// Function: sub_10E0940
// Address: 0x10e0940
//
__int64 __fastcall sub_10E0940(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  int v6; // ecx
  char v9; // dl
  int v10; // edx
  __int64 v11; // r12
  int v13; // edi
  __int64 v14; // rdx
  int v15; // r13d
  __int64 v16; // rbx
  __int64 v17; // r13
  __int64 v18; // rdx
  unsigned int v19; // esi
  __int64 v20; // rdx
  int v21; // r13d
  __int64 v22; // rbx
  __int64 v23; // r13
  __int64 v24; // rdx
  unsigned int v25; // esi
  __int64 v26; // rdx
  int v27; // r13d
  __int64 v28; // rbx
  __int64 v29; // r13
  __int64 v30; // rdx
  unsigned int v31; // esi
  _BYTE v32[32]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v33; // [rsp+20h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 8);
  if ( a3 == v4 )
    return a2;
  v6 = *(unsigned __int8 *)(v4 + 8);
  v9 = *(_BYTE *)(v4 + 8);
  if ( (unsigned int)(v6 - 17) > 1 )
  {
    if ( (_BYTE)v6 != 14 )
      goto LABEL_13;
  }
  else if ( *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL) != 14 )
  {
    goto LABEL_4;
  }
  v13 = *(unsigned __int8 *)(a3 + 8);
  if ( (unsigned int)(v13 - 17) <= 1 )
    LOBYTE(v13) = *(_BYTE *)(**(_QWORD **)(a3 + 16) + 8LL);
  if ( (_BYTE)v13 != 12 )
  {
LABEL_4:
    if ( v6 == 18 )
    {
LABEL_5:
      v9 = *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL);
LABEL_6:
      if ( v9 != 12 )
        goto LABEL_10;
      v10 = *(unsigned __int8 *)(a3 + 8);
      if ( (unsigned int)(v10 - 17) <= 1 )
        LOBYTE(v10) = *(_BYTE *)(**(_QWORD **)(a3 + 16) + 8LL);
      if ( (_BYTE)v10 == 14 )
      {
        v11 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a1[10] + 120LL))(
                a1[10],
                48,
                a2,
                a3);
        if ( !v11 )
        {
          v33 = 257;
          v11 = sub_B51D30(48, a2, a3, (__int64)v32, 0, 0);
          if ( (unsigned __int8)sub_920620(v11) )
          {
            v26 = a1[12];
            v27 = *((_DWORD *)a1 + 26);
            if ( v26 )
              sub_B99FD0(v11, 3u, v26);
            sub_B45150(v11, v27);
          }
          (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
            a1[11],
            v11,
            a4,
            a1[7],
            a1[8]);
          v28 = *a1;
          v29 = *a1 + 16LL * *((unsigned int *)a1 + 2);
          if ( *a1 != v29 )
          {
            do
            {
              v30 = *(_QWORD *)(v28 + 8);
              v31 = *(_DWORD *)v28;
              v28 += 16;
              sub_B99FD0(v11, v31, v30);
            }
            while ( v29 != v28 );
          }
        }
      }
      else
      {
LABEL_10:
        v11 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a1[10] + 120LL))(
                a1[10],
                49,
                a2,
                a3);
        if ( !v11 )
        {
          v33 = 257;
          v11 = sub_B51D30(49, a2, a3, (__int64)v32, 0, 0);
          if ( (unsigned __int8)sub_920620(v11) )
          {
            v20 = a1[12];
            v21 = *((_DWORD *)a1 + 26);
            if ( v20 )
              sub_B99FD0(v11, 3u, v20);
            sub_B45150(v11, v21);
          }
          (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
            a1[11],
            v11,
            a4,
            a1[7],
            a1[8]);
          v22 = *a1;
          v23 = *a1 + 16LL * *((unsigned int *)a1 + 2);
          if ( *a1 != v23 )
          {
            do
            {
              v24 = *(_QWORD *)(v22 + 8);
              v25 = *(_DWORD *)v22;
              v22 += 16;
              sub_B99FD0(v11, v25, v24);
            }
            while ( v23 != v22 );
          }
        }
      }
      return v11;
    }
LABEL_13:
    if ( v6 != 17 )
      goto LABEL_6;
    goto LABEL_5;
  }
  v11 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a1[10] + 120LL))(a1[10], 47, a2, a3);
  if ( !v11 )
  {
    v33 = 257;
    v11 = sub_B51D30(47, a2, a3, (__int64)v32, 0, 0);
    if ( (unsigned __int8)sub_920620(v11) )
    {
      v14 = a1[12];
      v15 = *((_DWORD *)a1 + 26);
      if ( v14 )
        sub_B99FD0(v11, 3u, v14);
      sub_B45150(v11, v15);
    }
    (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v11,
      a4,
      a1[7],
      a1[8]);
    v16 = *a1;
    v17 = *a1 + 16LL * *((unsigned int *)a1 + 2);
    if ( *a1 != v17 )
    {
      do
      {
        v18 = *(_QWORD *)(v16 + 8);
        v19 = *(_DWORD *)v16;
        v16 += 16;
        sub_B99FD0(v11, v19, v18);
      }
      while ( v17 != v16 );
    }
  }
  return v11;
}
