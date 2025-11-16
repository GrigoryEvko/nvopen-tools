// Function: sub_F005C0
// Address: 0xf005c0
//
__int64 __fastcall sub_F005C0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v10; // r13
  __int64 v13; // r8
  __int64 v14; // r9
  int v15; // esi
  __int64 v16; // rdi
  __int64 v17; // rax
  unsigned __int64 v18; // rcx
  __int64 v19; // rax
  unsigned __int64 v20; // rcx
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  unsigned int v25; // esi
  __int64 v26; // r8
  __int64 v27; // r9
  int v28; // eax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // [rsp-10h] [rbp-80h]
  __int64 v35; // [rsp-8h] [rbp-78h]
  unsigned int v36; // [rsp+38h] [rbp-38h]

  v10 = a1 + 1576;
  sub_A19830(a1 + 1576, 8u, 3u);
  v15 = *(_DWORD *)(a1 + 1060);
  v16 = a1 + 1048;
  v17 = 0;
  *(_DWORD *)(a1 + 1056) = 0;
  if ( !v15 )
  {
    sub_C8D5F0(v16, (const void *)(a1 + 1064), 1u, 8u, v13, v14);
    v16 = a1 + 1048;
    v17 = 8LL * *(unsigned int *)(a1 + 1056);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 1048) + v17) = 1;
  v18 = *(unsigned int *)(a1 + 1060);
  v19 = (unsigned int)(*(_DWORD *)(a1 + 1056) + 1);
  *(_DWORD *)(a1 + 1056) = v19;
  if ( v19 + 1 > v18 )
  {
    sub_C8D5F0(v16, (const void *)(a1 + 1064), v19 + 1, 8u, v13, v14);
    v19 = *(unsigned int *)(a1 + 1056);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 1048) + 8 * v19) = a2;
  v20 = *(unsigned int *)(a1 + 1060);
  v21 = *(int *)(a1 + 1728);
  v22 = (unsigned int)(*(_DWORD *)(a1 + 1056) + 1);
  *(_DWORD *)(a1 + 1056) = v22;
  if ( v22 + 1 > v20 )
  {
    sub_C8D5F0(v16, (const void *)(a1 + 1064), v22 + 1, 8u, v13, v14);
    v22 = *(unsigned int *)(a1 + 1056);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 1048) + 8 * v22) = v21;
  v23 = *(_QWORD *)(a1 + 1048);
  v24 = (unsigned int)(*(_DWORD *)(a1 + 1056) + 1);
  v25 = *(_DWORD *)(a1 + 1736);
  *(_DWORD *)(a1 + 1056) = v24;
  sub_EFE900(v10, v25, v23, v24, 0, 0, v36, 0);
  v28 = *(_DWORD *)(a1 + 1728);
  switch ( v28 )
  {
    case 1:
      sub_EFFC30(a1, a3, v34, v35, v26, v27);
      break;
    case 2:
      sub_EFFC30(a1, a3, v34, v35, v26, v27);
      sub_EFFA20(a1, a5, v30, v31, v32, v33);
      break;
    case 0:
      sub_EFFA20(a1, a5, v34, v35, v26, v27);
      sub_EFFB80(a1, a7, a8);
      break;
  }
  return sub_A192A0(v10);
}
