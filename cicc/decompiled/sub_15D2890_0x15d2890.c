// Function: sub_15D2890
// Address: 0x15d2890
//
void __fastcall sub_15D2890(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // r15
  unsigned int v10; // eax
  char *v11; // rdx
  __int64 *v12; // rax
  __int64 v13; // r15
  char **v14; // rbx
  __int64 v15; // r9
  char *v16; // r10
  __int64 v17; // rax
  char **v18; // rax
  char *v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rcx
  unsigned int v22; // edx
  __int64 v23; // rdi
  char *v24; // r9
  int v25; // edi
  int v26; // r8d
  __int64 v27; // [rsp+8h] [rbp-308h]
  __int64 v28; // [rsp+10h] [rbp-300h]
  __int64 v30; // [rsp+40h] [rbp-2D0h]
  __int64 *v31; // [rsp+48h] [rbp-2C8h]
  char *v32; // [rsp+48h] [rbp-2C8h]
  char **v34; // [rsp+60h] [rbp-2B0h]
  __int64 v35; // [rsp+68h] [rbp-2A8h] BYREF
  char *v36; // [rsp+70h] [rbp-2A0h] BYREF
  __int64 v37; // [rsp+78h] [rbp-298h] BYREF
  char **v38; // [rsp+80h] [rbp-290h] BYREF
  int v39; // [rsp+88h] [rbp-288h]
  char v40; // [rsp+90h] [rbp-280h] BYREF
  _QWORD *v41; // [rsp+D0h] [rbp-240h] BYREF
  __int64 v42; // [rsp+D8h] [rbp-238h]
  _QWORD v43[70]; // [rsp+E0h] [rbp-230h] BYREF

  v6 = a1 + 24;
  v35 = a2;
  v41 = v43;
  v43[0] = a2;
  v37 = a2;
  v42 = 0x4000000001LL;
  if ( (unsigned __int8)sub_15CE630(a1 + 24, &v37, &v38) )
    *((_DWORD *)sub_15D1D60(v6, &v35) + 3) = a6;
  v10 = v42;
  v28 = a5 + 16;
  if ( (_DWORD)v42 )
  {
LABEL_6:
    while ( 1 )
    {
      v11 = (char *)v41[v10 - 1];
      LODWORD(v42) = v10 - 1;
      v36 = v11;
      v12 = sub_15D1D60(v6, (__int64 *)&v36);
      if ( !*((_DWORD *)v12 + 2) )
        break;
LABEL_5:
      v10 = v42;
      if ( !(_DWORD)v42 )
        goto LABEL_23;
    }
    ++a3;
    v12[3] = (__int64)v36;
    *((_DWORD *)v12 + 4) = a3;
    *((_DWORD *)v12 + 2) = a3;
    sub_15CE600(a1, &v36);
    sub_15CF6C0((__int64)&v38, (__int64)v36, *(_QWORD *)(a1 + 56));
    v34 = &v38[v39];
    if ( v38 == v34 )
      goto LABEL_21;
    v30 = v6;
    v13 = a5;
    v14 = v38;
    while ( 1 )
    {
      v19 = *v14;
      v20 = *(unsigned int *)(a1 + 48);
      v37 = (__int64)*v14;
      if ( !(_DWORD)v20 )
        goto LABEL_9;
      v21 = *(_QWORD *)(a1 + 32);
      v22 = (v20 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v23 = v21 + 72LL * v22;
      v24 = *(char **)v23;
      if ( v19 != *(char **)v23 )
        break;
LABEL_16:
      if ( v23 == v21 + 72 * v20 || !*(_DWORD *)(v23 + 8) )
        goto LABEL_9;
      if ( v19 == v36 )
      {
LABEL_13:
        if ( v34 == ++v14 )
          goto LABEL_20;
      }
      else
      {
        ++v14;
        sub_15CDD90(v23 + 40, &v36);
        if ( v34 == v14 )
        {
LABEL_20:
          a5 = v13;
          v6 = v30;
          v34 = v38;
LABEL_21:
          if ( v34 == (char **)&v40 )
            goto LABEL_5;
          _libc_free((unsigned __int64)v34);
          v10 = v42;
          if ( !(_DWORD)v42 )
            goto LABEL_23;
          goto LABEL_6;
        }
      }
    }
    v25 = 1;
    while ( v24 != (char *)-8LL )
    {
      v26 = v25 + 1;
      v22 = (v20 - 1) & (v25 + v22);
      v23 = v21 + 72LL * v22;
      v24 = *(char **)v23;
      if ( v19 == *(char **)v23 )
        goto LABEL_16;
      v25 = v26;
    }
LABEL_9:
    v15 = sub_15CC510(a4, (__int64)v19);
    if ( v15 )
    {
      v16 = v36;
      v17 = *(unsigned int *)(v13 + 8);
      if ( (unsigned int)v17 >= *(_DWORD *)(v13 + 12) )
      {
        v27 = v15;
        v32 = v36;
        sub_16CD150(v13, v28, 0, 16);
        v17 = *(unsigned int *)(v13 + 8);
        v15 = v27;
        v16 = v32;
      }
      v18 = (char **)(*(_QWORD *)v13 + 16 * v17);
      *v18 = v16;
      v18[1] = (char *)v15;
      ++*(_DWORD *)(v13 + 8);
    }
    else
    {
      v31 = sub_15D1D60(v30, &v37);
      sub_15CDD90((__int64)&v41, &v37);
      *((_DWORD *)v31 + 3) = a3;
      sub_15CDD90((__int64)(v31 + 5), &v36);
    }
    goto LABEL_13;
  }
LABEL_23:
  if ( v41 != v43 )
    _libc_free((unsigned __int64)v41);
}
