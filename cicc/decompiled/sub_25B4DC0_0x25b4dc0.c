// Function: sub_25B4DC0
// Address: 0x25b4dc0
//
__int64 __fastcall sub_25B4DC0(__int64 a1, __int64 a2)
{
  char v2; // al
  unsigned int v3; // r14d
  unsigned int v4; // eax
  _QWORD **v5; // r13
  _QWORD **i; // r12
  __int64 v7; // rax
  _QWORD *v8; // rbx
  unsigned __int64 v9; // r15
  __int64 v10; // rdi
  unsigned int v11; // eax
  _QWORD *v12; // rbx
  _QWORD *v13; // r12
  __int64 v14; // rdi
  __int64 **v16; // rax
  __int64 **v17; // rdx
  _BYTE v18[8]; // [rsp+0h] [rbp-190h] BYREF
  _QWORD *v19; // [rsp+8h] [rbp-188h]
  unsigned int v20; // [rsp+18h] [rbp-178h]
  __int64 v21; // [rsp+28h] [rbp-168h]
  unsigned int v22; // [rsp+38h] [rbp-158h]
  __int64 v23; // [rsp+48h] [rbp-148h]
  unsigned int v24; // [rsp+58h] [rbp-138h]
  _BYTE v25[8]; // [rsp+60h] [rbp-130h] BYREF
  __int64 **v26; // [rsp+68h] [rbp-128h]
  int v27; // [rsp+74h] [rbp-11Ch]
  unsigned __int8 v28; // [rsp+7Ch] [rbp-114h]
  unsigned __int64 v29; // [rsp+98h] [rbp-F8h]
  int v30; // [rsp+A4h] [rbp-ECh]
  int v31; // [rsp+A8h] [rbp-E8h]
  char v32; // [rsp+ACh] [rbp-E4h]
  __int64 v33; // [rsp+C0h] [rbp-D0h] BYREF
  int v34; // [rsp+C8h] [rbp-C8h] BYREF
  unsigned __int64 v35; // [rsp+D0h] [rbp-C0h]
  int *v36; // [rsp+D8h] [rbp-B8h]
  int *v37; // [rsp+E0h] [rbp-B0h]
  __int64 v38; // [rsp+E8h] [rbp-A8h]
  int v39; // [rsp+F8h] [rbp-98h] BYREF
  unsigned __int64 v40; // [rsp+100h] [rbp-90h]
  int *v41; // [rsp+108h] [rbp-88h]
  int *v42; // [rsp+110h] [rbp-80h]
  __int64 v43; // [rsp+118h] [rbp-78h]
  int v44; // [rsp+128h] [rbp-68h] BYREF
  unsigned __int64 v45; // [rsp+130h] [rbp-60h]
  int *v46; // [rsp+138h] [rbp-58h]
  int *v47; // [rsp+140h] [rbp-50h]
  __int64 v48; // [rsp+148h] [rbp-48h]
  char v49; // [rsp+150h] [rbp-40h]

  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 152LL))(a1);
  v34 = 0;
  v35 = 0;
  v36 = &v34;
  v37 = &v34;
  v41 = &v39;
  v42 = &v39;
  v46 = &v44;
  v47 = &v44;
  v49 = v2;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v48 = 0;
  sub_BBB1A0((__int64)v18);
  v3 = 1;
  sub_25B4BE0((__int64)v25, &v33, a2);
  if ( v30 != v31 )
  {
LABEL_2:
    if ( v32 )
      goto LABEL_3;
    goto LABEL_32;
  }
  v3 = v28;
  if ( !v28 )
  {
    LOBYTE(v3) = sub_C8CA60((__int64)v25, (__int64)&qword_4F82400) == 0;
    goto LABEL_2;
  }
  v16 = v26;
  v17 = &v26[v27];
  if ( v26 != v17 )
  {
    while ( *v16 != &qword_4F82400 )
    {
      if ( v17 == ++v16 )
        goto LABEL_31;
    }
    v3 = 0;
  }
LABEL_31:
  if ( !v32 )
  {
LABEL_32:
    _libc_free(v29);
LABEL_3:
    if ( !v28 )
      _libc_free((unsigned __int64)v26);
  }
  sub_C7D6A0(v23, 24LL * v24, 8);
  v4 = v22;
  if ( v22 )
  {
    v5 = (_QWORD **)(v21 + 32LL * v22);
    for ( i = (_QWORD **)(v21 + 8); ; i += 4 )
    {
      v7 = (__int64)*(i - 1);
      if ( v7 != -4096 && v7 != -8192 )
      {
        v8 = *i;
        while ( v8 != i )
        {
          v9 = (unsigned __int64)v8;
          v8 = (_QWORD *)*v8;
          v10 = *(_QWORD *)(v9 + 24);
          if ( v10 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v10 + 8LL))(v10);
          j_j___libc_free_0(v9);
        }
      }
      if ( v5 == i + 3 )
        break;
    }
    v4 = v22;
  }
  sub_C7D6A0(v21, 32LL * v4, 8);
  v11 = v20;
  if ( v20 )
  {
    v12 = v19;
    v13 = &v19[2 * v20];
    do
    {
      if ( *v12 != -8192 && *v12 != -4096 )
      {
        v14 = v12[1];
        if ( v14 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 8LL))(v14);
      }
      v12 += 2;
    }
    while ( v13 != v12 );
    v11 = v20;
  }
  sub_C7D6A0((__int64)v19, 16LL * v11, 8);
  sub_25AE610(v45);
  sub_25AE270(v40);
  sub_25AE440(v35);
  return v3;
}
