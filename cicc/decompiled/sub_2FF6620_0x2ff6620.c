// Function: sub_2FF6620
// Address: 0x2ff6620
//
__int64 *__fastcall sub_2FF6620(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 **v3; // r14
  __int64 *v4; // r15
  __int64 v5; // rdx
  int v6; // esi
  unsigned int v7; // esi
  int v8; // edx
  __int64 *v9; // r12
  _WORD *v10; // rbx
  __int16 v11; // si
  __int64 *v12; // r12
  _WORD *v13; // r15
  unsigned int v16; // [rsp+Ch] [rbp-74h]
  __int64 v17; // [rsp+10h] [rbp-70h]
  __int64 *v18; // [rsp+18h] [rbp-68h]
  __int64 *v19; // [rsp+20h] [rbp-60h]
  unsigned int v20; // [rsp+28h] [rbp-58h]
  unsigned int v21; // [rsp+2Ch] [rbp-54h]
  __int64 **v22; // [rsp+30h] [rbp-50h]
  unsigned __int64 v23; // [rsp+38h] [rbp-48h]
  __int64 v24; // [rsp+48h] [rbp-38h] BYREF

  v23 = a3 & 0xFFFFFFFFFFFFFFF9LL;
  v22 = *(__int64 ***)(a1 + 288);
  if ( v22 != *(__int64 ***)(a1 + 280) )
  {
    v3 = *(__int64 ***)(a1 + 280);
    v21 = a2 - 1;
    v4 = &v24;
    v20 = a2 >> 3;
    v17 = a2 >> 3;
    v19 = 0;
    v16 = a2 & 7;
    while ( 1 )
    {
      v9 = *v3;
      if ( !v23 )
        break;
      v10 = (_WORD *)(*(_QWORD *)(a1 + 320)
                    + 2LL
                    * *(unsigned int *)(*(_QWORD *)(a1 + 312)
                                      + 16LL
                                      * (*(unsigned __int16 *)(*v9 + 24)
                                       + *(_DWORD *)(a1 + 328)
                                       * (unsigned int)((__int64)(*(_QWORD *)(a1 + 288) - *(_QWORD *)(a1 + 280)) >> 3))
                                      + 12));
      v11 = *v10;
      if ( *v10 == 1 )
      {
LABEL_10:
        if ( v22 == ++v3 )
          return v19;
      }
      else
      {
        v18 = *v3;
        v12 = v4;
        v13 = v10;
        do
        {
          if ( v11 == 264
            || (sub_34B2480(v12), (((unsigned __int8)a3 ^ (unsigned __int8)v24) & 7) == 0)
            && ((a3 ^ v24) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          {
            v4 = v12;
            v9 = v18;
            goto LABEL_3;
          }
          v11 = v13[1];
          ++v13;
        }
        while ( v11 != 1 );
        v4 = v12;
        if ( v22 == ++v3 )
          return v19;
      }
    }
LABEL_3:
    if ( v21 <= 0x3FFFFFFE )
    {
      v5 = *v9;
      if ( v20 < *(unsigned __int16 *)(*v9 + 22) )
      {
        v6 = *(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + v17);
        if ( _bittest(&v6, v16) )
        {
          if ( !v19
            || v9 != v19
            && (v7 = *(unsigned __int16 *)(v5 + 24),
                v8 = *(_DWORD *)(v19[1] + 4 * ((unsigned __int64)*(unsigned __int16 *)(v5 + 24) >> 5)),
                _bittest(&v8, v7)) )
          {
            v19 = v9;
          }
        }
      }
    }
    goto LABEL_10;
  }
  return 0;
}
