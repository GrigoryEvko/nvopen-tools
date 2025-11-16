// Function: sub_15A7010
// Address: 0x15a7010
//
__int64 __fastcall sub_15A7010(
        __int64 a1,
        _BYTE *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int a8,
        __int64 a9,
        unsigned __int8 a10,
        char a11,
        int a12,
        int a13,
        unsigned __int8 a14,
        __int64 a15,
        __int64 a16,
        __int64 a17)
{
  int v18; // r13d
  __int64 v20; // rax
  __int64 v21; // r9
  __int64 v22; // rdi
  int v23; // ecx
  int v24; // edx
  int v25; // eax
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // rdi
  int v29; // ecx
  int v30; // eax
  int v31; // edx
  int v32; // eax
  int v36; // [rsp+10h] [rbp-50h]
  __int64 v38; // [rsp+18h] [rbp-48h]
  int v39; // [rsp+18h] [rbp-48h]
  __int64 v40; // [rsp+20h] [rbp-40h]
  __int64 v41; // [rsp+28h] [rbp-38h]

  v18 = (int)a2;
  v20 = sub_1627350(*(_QWORD *)(a1 + 8), 0, 0, 2, 1);
  v21 = a6;
  v41 = v20;
  if ( a11 )
  {
    v22 = *(_QWORD *)(a1 + 8);
    v38 = *(_QWORD *)(a1 + 16);
    if ( a2 && *a2 == 16 )
      v18 = 0;
    v23 = 0;
    if ( v21 )
      v23 = sub_161FF10(v22, a5, v21);
    v24 = 0;
    if ( a4 )
    {
      v36 = v23;
      v25 = sub_161FF10(v22, a3, a4);
      v23 = v36;
      v24 = v25;
    }
    v26 = sub_15BFC70(v22, v18, v24, v23, a7, a8, a9, a10, 1, a12, 0, 0, 0, 0, a13, a14, v38, a15, a16, v41, a17, 1, 1);
    v27 = *(unsigned int *)(a1 + 152);
    if ( (unsigned int)v27 >= *(_DWORD *)(a1 + 156) )
    {
      sub_16CD150(a1 + 144, a1 + 160, 0, 8);
      v27 = *(unsigned int *)(a1 + 152);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 144) + 8 * v27) = v26;
    ++*(_DWORD *)(a1 + 152);
  }
  else
  {
    v28 = *(_QWORD *)(a1 + 8);
    if ( a2 && *a2 == 16 )
      v18 = 0;
    v29 = 0;
    if ( a6 )
    {
      v40 = *(_QWORD *)(a1 + 8);
      v30 = sub_161FF10(v28, a5, a6);
      v28 = v40;
      v29 = v30;
    }
    v31 = 0;
    if ( a4 )
    {
      v39 = v29;
      v32 = sub_161FF10(v28, a3, a4);
      v29 = v39;
      v31 = v32;
    }
    v26 = sub_15BFC70(v28, v18, v31, v29, a7, a8, a9, a10, 0, a12, 0, 0, 0, 0, a13, a14, 0, a15, a16, v41, a17, 0, 1);
  }
  sub_15A6B80(a1, v26);
  return v26;
}
