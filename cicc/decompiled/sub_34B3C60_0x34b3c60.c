// Function: sub_34B3C60
// Address: 0x34b3c60
//
__int64 __fastcall sub_34B3C60(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 result; // rax
  __int64 v5; // rbx
  __int64 v6; // r12
  unsigned int v7; // r12d
  __int64 v8; // rdx
  __int16 *v9; // r14
  __int64 v10; // rdx
  unsigned int v11; // r13d
  unsigned int v12; // esi
  __int64 v13; // rax
  bool v14; // r10
  __int64 v15; // rax
  __int64 v16; // [rsp+8h] [rbp-58h]
  char v18; // [rsp+1Ch] [rbp-44h]
  _QWORD *v19; // [rsp+20h] [rbp-40h]
  _QWORD *v20; // [rsp+20h] [rbp-40h]
  _QWORD *v21; // [rsp+28h] [rbp-38h]

  result = *(_DWORD *)(a2 + 40) & 0xFFFFFF;
  if ( (_DWORD)result )
  {
    v5 = 0;
    v16 = 40LL * (unsigned int)result;
    v21 = a3 + 1;
    while ( 1 )
    {
      result = a2;
      v6 = v5 + *(_QWORD *)(a2 + 32);
      if ( !*(_BYTE *)v6 )
      {
        if ( (*(_BYTE *)(v6 + 3) & 0x10) != 0 && (*(_WORD *)(v6 + 2) & 0xFF0) != 0
          || (result = sub_34B3BD0(a1, a2, v5 + *(_QWORD *)(a2 + 32)), (_BYTE)result) )
        {
          v7 = *(_DWORD *)(v6 + 8);
          v8 = *(_QWORD *)(a1 + 32);
          result = *(_QWORD *)(v8 + 56);
          v9 = (__int16 *)(result + 2LL * *(unsigned int *)(*(_QWORD *)(v8 + 8) + 24LL * v7 + 4));
          if ( v9 )
            break;
        }
      }
LABEL_22:
      v5 += 40;
      if ( v16 == v5 )
        return result;
    }
    v10 = a3[2];
    v11 = (unsigned __int16)v7;
    if ( v10 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v12 = *(_DWORD *)(v10 + 32);
          v13 = *(_QWORD *)(v10 + 24);
          if ( v11 < v12 )
            v13 = *(_QWORD *)(v10 + 16);
          if ( !v13 )
            break;
          v10 = v13;
        }
        if ( v11 < v12 )
          break;
        if ( v11 > v12 )
          goto LABEL_15;
LABEL_17:
        result = (unsigned int)*v9++;
        if ( !(_WORD)result )
          goto LABEL_22;
        v7 += result;
        v10 = a3[2];
        v11 = (unsigned __int16)v7;
        if ( !v10 )
          goto LABEL_19;
      }
      if ( v10 == a3[3] )
      {
LABEL_15:
        v14 = 1;
        if ( (_QWORD *)v10 != v21 )
LABEL_28:
          v14 = v11 < *(_DWORD *)(v10 + 32);
LABEL_16:
        v18 = v14;
        v19 = (_QWORD *)v10;
        v15 = sub_22077B0(0x28u);
        *(_DWORD *)(v15 + 32) = v11;
        sub_220F040(v18, v15, v19, v21);
        ++a3[5];
        goto LABEL_17;
      }
    }
    else
    {
LABEL_19:
      v10 = (__int64)v21;
      if ( (_QWORD *)a3[3] == v21 )
      {
        v10 = (__int64)v21;
        v14 = 1;
        goto LABEL_16;
      }
    }
    v20 = (_QWORD *)v10;
    if ( v11 <= *(_DWORD *)(sub_220EF80(v10) + 32) )
      goto LABEL_17;
    v10 = (__int64)v20;
    if ( !v20 )
      goto LABEL_17;
    v14 = 1;
    if ( v20 != v21 )
      goto LABEL_28;
    goto LABEL_16;
  }
  return result;
}
