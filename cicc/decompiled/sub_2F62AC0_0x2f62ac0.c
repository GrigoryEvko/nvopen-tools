// Function: sub_2F62AC0
// Address: 0x2f62ac0
//
void __fastcall sub_2F62AC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // r8
  __int64 v12; // rcx
  __int64 v13; // rax
  _QWORD **v14; // rax
  __int64 v15; // rbx
  _QWORD **v16; // rsi
  _QWORD **v17; // rdx
  _QWORD **v18; // rax
  __int64 v19; // rcx
  __int64 *v20; // rax
  _BYTE *v21; // [rsp+0h] [rbp-120h] BYREF
  __int64 v22; // [rsp+8h] [rbp-118h]
  _BYTE v23[32]; // [rsp+10h] [rbp-110h] BYREF
  _QWORD v24[3]; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v25; // [rsp+48h] [rbp-D8h]
  __int64 v26; // [rsp+50h] [rbp-D0h]
  __int64 v27; // [rsp+58h] [rbp-C8h]
  __int64 v28; // [rsp+60h] [rbp-C0h]
  __int64 v29; // [rsp+68h] [rbp-B8h]
  int v30; // [rsp+70h] [rbp-B0h]
  char v31; // [rsp+74h] [rbp-ACh]
  __int64 v32; // [rsp+78h] [rbp-A8h]
  __int64 v33; // [rsp+80h] [rbp-A0h]
  char *v34; // [rsp+88h] [rbp-98h]
  __int64 v35; // [rsp+90h] [rbp-90h]
  int v36; // [rsp+98h] [rbp-88h]
  char v37; // [rsp+9Ch] [rbp-84h]
  char v38; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v39; // [rsp+C0h] [rbp-60h]
  char *v40; // [rsp+C8h] [rbp-58h]
  __int64 v41; // [rsp+D0h] [rbp-50h]
  int v42; // [rsp+D8h] [rbp-48h]
  char v43; // [rsp+DCh] [rbp-44h]
  char v44; // [rsp+E0h] [rbp-40h] BYREF

  v22 = 0x800000000LL;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = *(_QWORD *)(a1 + 40);
  v24[2] = &v21;
  v9 = *(_QWORD *)(v7 + 32);
  v27 = 0;
  v26 = v8;
  v10 = 0;
  v25 = v9;
  v11 = *(_QWORD *)(v7 + 16);
  v21 = v23;
  v24[0] = &unk_4A388F0;
  v24[1] = 0;
  v12 = *(_QWORD *)(*(_QWORD *)v11 + 128LL);
  v13 = 0;
  if ( (__int64 (*)())v12 != sub_2DAC790 )
  {
    v13 = ((__int64 (__fastcall *)(__int64, void *, _QWORD))v12)(v11, &unk_4A388F0, 0);
    v10 = (unsigned int)v22;
    v9 = v25;
  }
  v28 = v13;
  v34 = &v38;
  v29 = a1;
  v30 = v10;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v35 = 4;
  v36 = 0;
  v37 = 1;
  v39 = 0;
  v40 = &v44;
  v41 = 4;
  v42 = 0;
  v43 = 1;
  if ( !*(_BYTE *)(v9 + 36) )
    goto LABEL_20;
  v14 = *(_QWORD ***)(v9 + 16);
  v12 = *(unsigned int *)(v9 + 28);
  v10 = (__int64)&v14[v12];
  if ( v14 == (_QWORD **)v10 )
  {
LABEL_19:
    if ( (unsigned int)v12 >= *(_DWORD *)(v9 + 24) )
    {
LABEL_20:
      sub_C8CC70(v9 + 8, (__int64)v24, v10, v12, v11, a6);
      goto LABEL_8;
    }
    *(_DWORD *)(v9 + 28) = v12 + 1;
    *(_QWORD *)v10 = v24;
    ++*(_QWORD *)(v9 + 8);
  }
  else
  {
    while ( *v14 != v24 )
    {
      if ( (_QWORD **)v10 == ++v14 )
        goto LABEL_19;
    }
  }
LABEL_8:
  sub_350D230(v24, a1 + 760, 0, 0);
  v15 = v25;
  v24[0] = &unk_4A388F0;
  if ( *(_BYTE *)(v25 + 36) )
  {
    v16 = *(_QWORD ***)(v25 + 16);
    v17 = &v16[*(unsigned int *)(v25 + 28)];
    v18 = v16;
    if ( v16 != v17 )
    {
      while ( *v18 != v24 )
      {
        if ( v17 == ++v18 )
          goto LABEL_14;
      }
      v19 = (unsigned int)(*(_DWORD *)(v25 + 28) - 1);
      *(_DWORD *)(v25 + 28) = v19;
      *v18 = v16[v19];
      ++*(_QWORD *)(v15 + 8);
    }
LABEL_14:
    if ( v43 )
      goto LABEL_15;
LABEL_23:
    _libc_free((unsigned __int64)v40);
    if ( v37 )
      goto LABEL_16;
    goto LABEL_24;
  }
  v20 = sub_C8CA60(v25 + 8, (__int64)v24);
  if ( !v20 )
    goto LABEL_14;
  *v20 = -2;
  ++*(_DWORD *)(v15 + 32);
  ++*(_QWORD *)(v15 + 8);
  if ( !v43 )
    goto LABEL_23;
LABEL_15:
  if ( v37 )
    goto LABEL_16;
LABEL_24:
  _libc_free((unsigned __int64)v34);
LABEL_16:
  if ( v21 != v23 )
    _libc_free((unsigned __int64)v21);
}
