// Function: sub_2E9AFF0
// Address: 0x2e9aff0
//
__int64 __fastcall sub_2E9AFF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  char v5; // al
  __int64 result; // rax
  int v8; // eax
  __int64 v9; // rax
  int v10; // eax
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 (*v13)(); // rcx
  __int64 v14; // rax
  int *v15; // rdx
  int *v16; // rcx
  int v17; // eax
  __int64 **v18; // rdx
  __int64 v19; // rcx
  __int64 **i; // rcx
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  int v23; // eax
  __int64 v24; // [rsp+8h] [rbp-28h] BYREF
  char v25[17]; // [rsp+1Fh] [rbp-11h] BYREF

  v5 = 1;
  v24 = a3;
  if ( byte_50205C8 )
    v5 = *(_BYTE *)sub_2E9AD00(a1 + 400, &v24) ^ 1;
  v25[0] = v5;
  if ( !(unsigned __int8)sub_2E8B400(a2, (__int64)v25, a3, a4, a5)
    && (!(_BYTE)qword_50206A8 || !(unsigned __int8)sub_2E986C0(a2, *(_QWORD *)(a1 + 16), *(_QWORD *)(a1 + 32))) )
  {
    return 0;
  }
  if ( (unsigned int)*(unsigned __int16 *)(a2 + 68) - 1 <= 1 && (*(_BYTE *)(*(_QWORD *)(a2 + 32) + 64LL) & 8) != 0
    || ((v8 = *(_DWORD *)(a2 + 44), (v8 & 4) == 0) && (v8 & 8) != 0
      ? (LOBYTE(v9) = sub_2E88A90(a2, 0x80000, 1))
      : (v9 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 19) & 1LL),
        (_BYTE)v9) )
  {
    v14 = *(_QWORD *)(a2 + 48);
    v15 = (int *)(v14 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v14 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      if ( (v14 & 7) != 0 )
      {
        if ( (v14 & 7) != 3 || !*v15 )
          goto LABEL_14;
        v16 = (int *)(*(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL);
      }
      else
      {
        *(_QWORD *)(a2 + 48) = v15;
        LOBYTE(v14) = v14 & 0xF8;
        v16 = v15;
      }
      v17 = v14 & 7;
      if ( v17 )
      {
        if ( v17 != 3 )
          goto LABEL_34;
        v18 = (__int64 **)(v16 + 4);
        v19 = *v16;
      }
      else
      {
        *(_QWORD *)(a2 + 48) = v16;
        v18 = (__int64 **)(a2 + 48);
        v19 = 1;
      }
      for ( i = &v18[v19]; i != v18; ++v18 )
      {
        v21 = **v18;
        if ( v21 )
        {
          if ( (v21 & 4) != 0 )
          {
            v22 = v21 & 0xFFFFFFFFFFFFFFF8LL;
            if ( v22 )
            {
              if ( (*(_DWORD *)(v22 + 8) & 0xFFFFFFFD) == 1 )
                goto LABEL_14;
            }
          }
        }
      }
LABEL_34:
      v23 = *(_DWORD *)(a1 + 1456);
      if ( v23 == 2 )
      {
        if ( (unsigned __int8)sub_2E97850(a1, *(_QWORD *)(a2 + 24), v24) )
          goto LABEL_14;
      }
      else if ( !v23 )
      {
        goto LABEL_14;
      }
      return 0;
    }
  }
LABEL_14:
  if ( (unsigned int)*(unsigned __int16 *)(a2 + 68) - 1 <= 1 && (*(_BYTE *)(*(_QWORD *)(a2 + 32) + 64LL) & 0x20) != 0 )
    return 0;
  v10 = *(_DWORD *)(a2 + 44);
  if ( (v10 & 0x20000) == 0 )
  {
    if ( (v10 & 4) != 0 || (v10 & 8) == 0 )
      v11 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 36) & 1LL;
    else
      LOBYTE(v11) = sub_2E88A90(a2, 0x1000000000LL, 1);
    if ( (_BYTE)v11 )
      return 0;
  }
  v12 = *(_QWORD *)a1;
  v13 = *(__int64 (**)())(**(_QWORD **)a1 + 208LL);
  result = 1;
  if ( v13 != sub_2E97340 )
    return ((__int64 (__fastcall *)(__int64, __int64, __int64))v13)(v12, a2, v24);
  return result;
}
