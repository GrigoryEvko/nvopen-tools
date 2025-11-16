// Function: sub_1480BD0
// Address: 0x1480bd0
//
__int64 __fastcall sub_1480BD0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 *v7; // rbx
  __int64 *v8; // rbx
  __int64 *v9; // rbx
  __int64 *v10; // r14
  unsigned __int8 v11; // al
  __int64 *v12; // rbx
  __int64 *v13; // rax
  __int64 *v14; // rax
  unsigned __int8 v15; // al
  __int64 v16; // rax
  unsigned __int8 v17; // [rsp+10h] [rbp-100h]
  unsigned __int8 v18; // [rsp+10h] [rbp-100h]
  unsigned __int8 v19; // [rsp+10h] [rbp-100h]
  __int64 *v20; // [rsp+10h] [rbp-100h]
  unsigned __int8 v21; // [rsp+20h] [rbp-F0h]
  unsigned __int8 v22; // [rsp+20h] [rbp-F0h]
  __int64 *v23; // [rsp+20h] [rbp-F0h]
  unsigned __int8 v24; // [rsp+38h] [rbp-D8h]
  __int64 v25[2]; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v26[2]; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v27[2]; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v28[2]; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v29; // [rsp+80h] [rbp-90h] BYREF
  unsigned int v30; // [rsp+88h] [rbp-88h]
  __int64 v31; // [rsp+90h] [rbp-80h] BYREF
  unsigned int v32; // [rsp+98h] [rbp-78h]
  __int64 v33; // [rsp+A0h] [rbp-70h] BYREF
  unsigned int v34; // [rsp+A8h] [rbp-68h]
  __int64 v35; // [rsp+B0h] [rbp-60h] BYREF
  unsigned int v36; // [rsp+B8h] [rbp-58h]
  __int64 v37; // [rsp+C0h] [rbp-50h] BYREF
  unsigned int v38; // [rsp+C8h] [rbp-48h]
  __int64 v39; // [rsp+D0h] [rbp-40h] BYREF
  unsigned int v40; // [rsp+D8h] [rbp-38h]

  result = sub_1452FA0(a3, a4);
  if ( (_BYTE)result )
    return sub_15FF820(a2);
  if ( a2 != 32 )
  {
    if ( a2 == 33 )
    {
      v9 = sub_1477920(a1, a4, 1u);
      sub_13A38D0((__int64)v27, (__int64)v9);
      sub_13A38D0((__int64)v28, (__int64)(v9 + 2));
      v10 = sub_1477920(a1, a3, 1u);
      sub_13A38D0((__int64)v25, (__int64)v10);
      sub_13A38D0((__int64)v26, (__int64)(v10 + 2));
      sub_1590F80(&v37, 33, v27);
      v21 = sub_158BB40(&v37, v25);
      sub_135E100(&v39);
      sub_135E100(&v37);
      v11 = v21;
      if ( !v21 )
      {
        v13 = sub_1477920(a1, a4, 0);
        v34 = *((_DWORD *)v13 + 2);
        if ( v34 > 0x40 )
        {
          v23 = v13;
          sub_16A4FD0(&v33, v13);
          v13 = v23;
        }
        else
        {
          v33 = *v13;
        }
        v36 = *((_DWORD *)v13 + 6);
        if ( v36 > 0x40 )
          sub_16A4FD0(&v35, v13 + 2);
        else
          v35 = v13[2];
        v14 = sub_1477920(a1, a3, 0);
        v30 = *((_DWORD *)v14 + 2);
        if ( v30 > 0x40 )
        {
          v20 = v14;
          sub_16A4FD0(&v29, v14);
          v14 = v20;
        }
        else
        {
          v29 = *v14;
        }
        v32 = *((_DWORD *)v14 + 6);
        if ( v32 > 0x40 )
          sub_16A4FD0(&v31, v14 + 2);
        else
          v31 = v14[2];
        sub_1590F80(&v37, 33, &v33);
        v15 = sub_158BB40(&v37, &v29);
        if ( v40 > 0x40 && v39 )
        {
          v17 = v15;
          j_j___libc_free_0_0(v39);
          v15 = v17;
        }
        if ( v38 > 0x40 && v37 )
        {
          v18 = v15;
          j_j___libc_free_0_0(v37);
          v15 = v18;
        }
        if ( !v15 )
        {
          v16 = sub_14806B0(a1, a3, a4, 0, 0);
          v15 = sub_1477CE0(a1, v16);
        }
        v19 = v15;
        sub_135E100(&v31);
        sub_135E100(&v29);
        sub_135E100(&v35);
        sub_135E100(&v33);
        v11 = v19;
      }
      v22 = v11;
      sub_135E100(v26);
      sub_135E100(v25);
      sub_135E100(v28);
      sub_135E100(v27);
      return v22;
    }
    else
    {
      if ( (unsigned __int8)sub_15FF7F0(a2) )
      {
        v12 = sub_1477920(a1, a4, 1u);
        sub_13A38D0((__int64)&v33, (__int64)v12);
        sub_13A38D0((__int64)&v35, (__int64)(v12 + 2));
        v8 = sub_1477920(a1, a3, 1u);
      }
      else
      {
        v7 = sub_1477920(a1, a4, 0);
        sub_13A38D0((__int64)&v33, (__int64)v7);
        sub_13A38D0((__int64)&v35, (__int64)(v7 + 2));
        v8 = sub_1477920(a1, a3, 0);
      }
      sub_13A38D0((__int64)&v29, (__int64)v8);
      sub_13A38D0((__int64)&v31, (__int64)(v8 + 2));
      sub_1590F80(&v37, a2, &v33);
      v24 = sub_158BB40(&v37, &v29);
      sub_135E100(&v39);
      sub_135E100(&v37);
      sub_135E100(&v31);
      sub_135E100(&v29);
      sub_135E100(&v35);
      sub_135E100(&v33);
      return v24;
    }
  }
  return result;
}
