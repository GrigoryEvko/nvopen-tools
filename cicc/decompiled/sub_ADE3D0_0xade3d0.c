// Function: sub_ADE3D0
// Address: 0xade3d0
//
__int64 __fastcall sub_ADE3D0(
        __int64 a1,
        _BYTE *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int a8,
        __int64 a9,
        int a10,
        int a11,
        int a12,
        __int64 a13,
        __int64 a14,
        __int64 a15,
        __int64 a16,
        __int64 a17,
        __int64 a18)
{
  __int64 v18; // r10
  int v20; // r12d
  __int64 v21; // r14
  __int64 v22; // rbx
  __int64 v23; // rax
  int v24; // ecx
  int v25; // eax
  int v26; // edx
  int v27; // eax
  __int64 v28; // r12
  __int64 v29; // rax
  __int64 v30; // rbx
  __int64 v31; // rax
  int v32; // ecx
  int v33; // eax
  int v34; // edx
  int v35; // eax
  __int64 v37; // [rsp+0h] [rbp-50h]
  __int64 v39; // [rsp+8h] [rbp-48h]
  int v40; // [rsp+8h] [rbp-48h]
  __int64 v41; // [rsp+8h] [rbp-48h]
  __int64 v42; // [rsp+10h] [rbp-40h]
  __int64 v44; // [rsp+10h] [rbp-40h]
  int v45; // [rsp+10h] [rbp-40h]

  v18 = a4;
  v20 = (int)a2;
  v21 = *(_QWORD *)(a1 + 8);
  if ( (a12 & 8) != 0 )
  {
    v42 = *(_QWORD *)(a1 + 16);
    if ( a2 && *a2 == 17 )
      v20 = 0;
    v22 = 0;
    if ( a18 )
    {
      v37 = a5;
      v23 = sub_B9B140(v21, a17, a18);
      a5 = v37;
      v18 = a4;
      v22 = v23;
    }
    v24 = 0;
    if ( a6 )
    {
      v39 = v18;
      v25 = sub_B9B140(v21, a5, a6);
      v18 = v39;
      v24 = v25;
    }
    v26 = 0;
    if ( v18 )
    {
      v40 = v24;
      v27 = sub_B9B140(v21, a3, v18);
      v24 = v40;
      v26 = v27;
    }
    v28 = sub_B07EA0(v21, v20, v26, v24, a7, a8, a9, a10, 0, 0, 0, a11, a12, v42, a13, a14, 0, a15, a16, v22, 1, 1);
    v29 = *(unsigned int *)(a1 + 160);
    if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 164) )
    {
      sub_C8D5F0(a1 + 152, a1 + 168, v29 + 1, 8);
      v29 = *(unsigned int *)(a1 + 160);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 152) + 8 * v29) = v28;
    ++*(_DWORD *)(a1 + 160);
  }
  else
  {
    if ( a2 && *a2 == 17 )
      v20 = 0;
    v30 = 0;
    if ( a18 )
    {
      v41 = a5;
      v31 = sub_B9B140(v21, a17, a18);
      a5 = v41;
      v18 = a4;
      v30 = v31;
    }
    v32 = 0;
    if ( a6 )
    {
      v44 = v18;
      v33 = sub_B9B140(v21, a5, a6);
      v18 = v44;
      v32 = v33;
    }
    v34 = 0;
    if ( v18 )
    {
      v45 = v32;
      v35 = sub_B9B140(v21, a3, v18);
      v32 = v45;
      v34 = v35;
    }
    v28 = sub_B07EA0(v21, v20, v34, v32, a7, a8, a9, a10, 0, 0, 0, a11, a12, 0, a13, a14, 0, a15, a16, v30, 0, 1);
  }
  sub_ADDDC0(a1, v28);
  return v28;
}
