// Function: sub_F02520
// Address: 0xf02520
//
__int64 __fastcall sub_F02520(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // al
  const char *v8; // r12
  char *v9; // r8
  int v10; // ecx
  unsigned int v11; // r15d
  int v12; // ebx
  unsigned int v13; // ecx
  int v14; // ebx
  _QWORD *v15; // r14
  __int64 v16; // rax
  unsigned int v17; // eax
  int v18; // edx
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  _QWORD *v24; // rax
  _QWORD *v25; // rdi
  _QWORD *v26; // r14
  __int64 v27; // r15
  __int64 v28; // r12
  volatile signed __int32 *v29; // rbx
  signed __int32 v30; // eax
  signed __int32 v31; // eax
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  int v36; // eax
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  _QWORD *v54; // [rsp+8h] [rbp-48h]

  v6 = 82;
  v8 = "RMRK";
  v9 = "";
  while ( 1 )
  {
    v10 = *(_DWORD *)(a1 + 1624);
    v11 = v6;
    v12 = v6 << v10;
    v13 = v10 + 8;
    v14 = *(_DWORD *)(a1 + 1628) | v12;
    *(_DWORD *)(a1 + 1628) = v14;
    if ( v13 > 0x1F )
      break;
    ++v8;
    *(_DWORD *)(a1 + 1624) = v13;
    if ( v9 == v8 )
      goto LABEL_10;
LABEL_4:
    v6 = *v8;
  }
  v15 = *(_QWORD **)(a1 + 1600);
  v16 = v15[1];
  if ( (unsigned __int64)(v16 + 4) > v15[2] )
  {
    sub_C8D290(*(_QWORD *)(a1 + 1600), v15 + 3, v16 + 4, 1u, (__int64)v9, a6);
    v16 = v15[1];
    v9 = "";
  }
  *(_DWORD *)(*v15 + v16) = v14;
  v17 = 0;
  v15[1] += 4LL;
  v18 = *(_DWORD *)(a1 + 1624);
  if ( v18 )
    v17 = v11 >> (32 - v18);
  ++v8;
  *(_DWORD *)(a1 + 1628) = v17;
  *(_DWORD *)(a1 + 1624) = ((_BYTE)v18 + 8) & 0x1F;
  if ( v9 != v8 )
    goto LABEL_4;
LABEL_10:
  v19 = 0;
  sub_A19830(a1 + 1576, 0, 2u);
  v24 = *(_QWORD **)(a1 + 1704);
  v25 = *(_QWORD **)(a1 + 1712);
  *(_DWORD *)(a1 + 1636) = -1;
  v54 = v24;
  if ( v24 != v25 )
  {
    v26 = v24;
    do
    {
      v27 = v26[2];
      v28 = v26[1];
      if ( v27 != v28 )
      {
        do
        {
          while ( 1 )
          {
            v29 = *(volatile signed __int32 **)(v28 + 8);
            if ( v29 )
            {
              if ( &_pthread_key_create )
              {
                v30 = _InterlockedExchangeAdd(v29 + 2, 0xFFFFFFFF);
              }
              else
              {
                v30 = *((_DWORD *)v29 + 2);
                v19 = (unsigned int)(v30 - 1);
                *((_DWORD *)v29 + 2) = v19;
              }
              if ( v30 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v29 + 16LL))(v29);
                if ( &_pthread_key_create )
                {
                  v31 = _InterlockedExchangeAdd(v29 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v31 = *((_DWORD *)v29 + 3);
                  v19 = (unsigned int)(v31 - 1);
                  *((_DWORD *)v29 + 3) = v19;
                }
                if ( v31 == 1 )
                  break;
              }
            }
            v28 += 16;
            if ( v27 == v28 )
              goto LABEL_23;
          }
          v28 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v29 + 24LL))(v29);
        }
        while ( v27 != v28 );
LABEL_23:
        v28 = v26[1];
      }
      if ( v28 )
      {
        v19 = v26[3] - v28;
        j_j___libc_free_0(v28, v19);
      }
      v26 += 4;
    }
    while ( v25 != v26 );
    *(_QWORD *)(a1 + 1712) = v54;
  }
  sub_F007D0(a1, v19, v20, v21, v22, v23);
  v36 = *(_DWORD *)(a1 + 1728);
  switch ( v36 )
  {
    case 1:
      sub_F00B10(a1, v19, v32, v33, v34, v35);
      sub_F01350(a1, v19, v38, v39, v40, v41);
      break;
    case 2:
      sub_F00B10(a1, v19, v32, v33, v34, v35);
      sub_F00DD0(a1, v19, v46, v47, v48, v49);
      sub_F01350(a1, v19, v50, v51, v52, v53);
      break;
    case 0:
      sub_F00DD0(a1, v19, v32, v33, v34, v35);
      sub_F01090(a1, v19, v42, v43, v44, v45);
      break;
  }
  return sub_A192A0(a1 + 1576);
}
