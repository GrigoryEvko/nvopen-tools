// Function: sub_87F190
// Address: 0x87f190
//
_QWORD *__fastcall sub_87F190(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  _QWORD *v7; // r12
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD *v11; // rcx
  __int64 v13; // r15
  __int64 **v14; // rax
  __int64 v15; // rdx
  __int64 *v16; // rdx
  __int64 *v17; // r8
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v22; // [rsp+8h] [rbp-38h]
  _QWORD *v23; // [rsp+8h] [rbp-38h]

  v22 = *(_QWORD *)(*(_QWORD *)(a2 + 168) + 152LL);
  v7 = sub_87EBB0(0x10u, *(_QWORD *)a1, (_QWORD *)(a1 + 48));
  sub_877E20((__int64)v7, 0, a2, v8, v9, v10);
  if ( v22 && (*(_BYTE *)(v22 + 29) & 0x20) == 0 )
    *((_DWORD *)v7 + 10) = *(_DWORD *)(v22 + 24);
  v11 = (_QWORD *)v7[11];
  *((_BYTE *)v7 + 82) = (4 * (a5 & 1)) | *((_BYTE *)v7 + 82) & 0xFB;
  if ( *(_BYTE *)(a1 + 80) == 16 )
  {
    *v11 = **(_QWORD **)(a1 + 88);
    if ( (*(_BYTE *)(a1 + 96) & 0xC) != 0 )
    {
      *((_BYTE *)v7 + 96) |= 8u;
      if ( a3 )
        goto LABEL_7;
      goto LABEL_11;
    }
  }
  else
  {
    *v11 = a1;
  }
  if ( a3 )
  {
LABEL_7:
    v11[1] = a3;
    return v7;
  }
LABEL_11:
  v13 = *(_QWORD *)(*v11 + 64LL);
  if ( a5 )
    goto LABEL_22;
  v14 = **(__int64 ****)(a2 + 168);
  if ( v14 )
  {
    while ( 1 )
    {
      v16 = v14[5];
      if ( v16 == (__int64 *)v13 )
        break;
      if ( v13 )
      {
        if ( v16 )
        {
          if ( dword_4F07588 )
          {
            v15 = v16[4];
            if ( *(_QWORD *)(v13 + 32) == v15 )
            {
              if ( v15 )
                break;
            }
          }
        }
      }
      v14 = (__int64 **)*v14;
      if ( !v14 )
        return v7;
    }
    v11[1] = v14;
    if ( ((_BYTE)v14[12] & 4) != 0 )
    {
LABEL_22:
      v17 = *(__int64 **)(a4 + 16);
      v18 = v17[7];
      if ( v18 != a2 )
      {
        if ( !v18 || !dword_4F07588 || (v20 = *(_QWORD *)(v18 + 32), *(_QWORD *)(a2 + 32) != v20) || !v20 )
        {
          v23 = v11;
          v19 = sub_8E5310(*(_QWORD *)(a4 + 16), a2, 0);
          v11 = v23;
          v17 = (__int64 *)v19;
        }
      }
      v11[1] = sub_8793F0(v13, a2, v17);
    }
  }
  return v7;
}
