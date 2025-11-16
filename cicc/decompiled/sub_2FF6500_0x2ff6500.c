// Function: sub_2FF6500
// Address: 0x2ff6500
//
__int64 *__fastcall sub_2FF6500(__int64 a1, unsigned int a2, __int16 a3)
{
  __int64 **v3; // r9
  __int64 **v5; // rsi
  unsigned int v6; // r12d
  unsigned int v8; // r13d
  __int64 *v9; // r10
  unsigned int v10; // ebx
  __int64 v11; // r11
  __int64 *v12; // r8
  __int16 *v13; // rdx
  __int16 v14; // ax
  __int64 v15; // rax
  int v16; // edx
  unsigned int v17; // edx
  int v18; // eax

  v3 = *(__int64 ***)(a1 + 288);
  v5 = *(__int64 ***)(a1 + 280);
  if ( v3 != v5 )
  {
    v6 = a2 - 1;
    v8 = a2 >> 3;
    v9 = 0;
    v10 = a2 & 7;
    v11 = v3 - v5;
    while ( 1 )
    {
      v12 = *v5;
      if ( a3 == 1 )
        goto LABEL_8;
      v13 = (__int16 *)(*(_QWORD *)(a1 + 320)
                      + 2LL
                      * *(unsigned int *)(*(_QWORD *)(a1 + 312)
                                        + 16LL
                                        * ((_DWORD)v11 * *(_DWORD *)(a1 + 328)
                                         + (unsigned int)*(unsigned __int16 *)(*v12 + 24))
                                        + 12));
      v14 = *v13;
      if ( *v13 != 1 )
        break;
LABEL_15:
      if ( v3 == ++v5 )
        return v9;
    }
    while ( a3 != v14 )
    {
      v14 = v13[1];
      ++v13;
      if ( v14 == 1 )
        goto LABEL_15;
    }
LABEL_8:
    if ( v6 <= 0x3FFFFFFE )
    {
      v15 = *v12;
      if ( v8 < *(unsigned __int16 *)(*v12 + 22) )
      {
        v16 = *(unsigned __int8 *)(*(_QWORD *)(v15 + 8) + v8);
        if ( _bittest(&v16, v10) )
        {
          if ( !v9
            || v12 != v9
            && (v17 = *(unsigned __int16 *)(v15 + 24),
                v18 = *(_DWORD *)(v9[1] + 4 * ((unsigned __int64)*(unsigned __int16 *)(v15 + 24) >> 5)),
                _bittest(&v18, v17)) )
          {
            v9 = *v5;
          }
        }
      }
    }
    goto LABEL_15;
  }
  return 0;
}
