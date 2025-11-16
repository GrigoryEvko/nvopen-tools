// Function: sub_AFC940
// Address: 0xafc940
//
__int64 __fastcall sub_AFC940(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // r14d
  __int64 v6; // rbx
  _BYTE *v8; // r15
  __int64 v9; // rax
  _BYTE *v10; // rdi
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 *v14; // rdx
  unsigned int v15; // eax
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 *v18; // rdx
  unsigned int v19; // eax
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 *v22; // rdx
  unsigned int v23; // eax
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 *v26; // rdx
  unsigned int v27; // eax
  __int64 v28; // rcx
  int v29; // r14d
  int v30; // eax
  __int64 v31; // rsi
  unsigned int v32; // eax
  _QWORD *v33; // rdi
  int v34; // r8d
  _QWORD *v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // [rsp+8h] [rbp-B8h]
  int v38; // [rsp+1Ch] [rbp-A4h] BYREF
  __int64 v39; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v40; // [rsp+28h] [rbp-98h] BYREF
  __int64 v41; // [rsp+30h] [rbp-90h] BYREF
  __int64 v42; // [rsp+38h] [rbp-88h] BYREF
  int v43; // [rsp+40h] [rbp-80h] BYREF
  __int64 v44[2]; // [rsp+48h] [rbp-78h] BYREF
  int v45; // [rsp+58h] [rbp-68h]
  int v46; // [rsp+5Ch] [rbp-64h] BYREF
  __int64 v47[2]; // [rsp+60h] [rbp-60h] BYREF
  __int64 v48; // [rsp+70h] [rbp-50h]
  __int64 v49; // [rsp+78h] [rbp-48h]
  __int64 v50; // [rsp+80h] [rbp-40h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v37 = *(_QWORD *)(a1 + 8);
    v8 = (_BYTE *)(*a2 - 16);
    v41 = sub_AF5140(*a2, 2u);
    v9 = v6;
    if ( *(_BYTE *)v6 != 16 )
      v9 = *(_QWORD *)sub_A17150((_BYTE *)(v6 - 16));
    v42 = v9;
    v43 = *(_DWORD *)(v6 + 16);
    v44[0] = *((_QWORD *)sub_A17150((_BYTE *)(v6 - 16)) + 1);
    v44[1] = *(_QWORD *)(v6 + 24);
    v45 = sub_AF18D0(v6);
    v46 = *(_DWORD *)(v6 + 20);
    v47[0] = *((_QWORD *)sub_A17150((_BYTE *)(v6 - 16)) + 3);
    v10 = (_BYTE *)(v6 - 16);
    v11 = *((_QWORD *)sub_A17150((_BYTE *)(v6 - 16)) + 4);
    v47[1] = v11;
    v48 = *((_QWORD *)sub_A17150(v10) + 5);
    v49 = *((_QWORD *)sub_A17150(v8) + 6);
    v38 = 0;
    v12 = *((_QWORD *)sub_A17150(v8) + 7);
    v39 = v11;
    v50 = v12;
    if ( v11 && *(_BYTE *)v11 == 1 )
    {
      v13 = *(_QWORD *)(v11 + 136);
      v14 = *(__int64 **)(v13 + 24);
      v15 = *(_DWORD *)(v13 + 32);
      if ( v15 > 0x40 )
      {
        v16 = *v14;
      }
      else
      {
        v16 = 0;
        if ( v15 )
          v16 = (__int64)((_QWORD)v14 << (64 - (unsigned __int8)v15)) >> (64 - (unsigned __int8)v15);
      }
      v40 = v16;
      v38 = sub_AF6E60(&v38, &v40);
    }
    else
    {
      v38 = sub_AF7970(&v38, &v39);
    }
    v39 = v48;
    if ( v48 && *(_BYTE *)v48 == 1 )
    {
      v17 = *(_QWORD *)(v48 + 136);
      v18 = *(__int64 **)(v17 + 24);
      v19 = *(_DWORD *)(v17 + 32);
      if ( v19 > 0x40 )
      {
        v20 = *v18;
      }
      else
      {
        v20 = 0;
        if ( v19 )
          v20 = (__int64)((_QWORD)v18 << (64 - (unsigned __int8)v19)) >> (64 - (unsigned __int8)v19);
      }
      v40 = v20;
      v38 = sub_AF6E60(&v38, &v40);
    }
    else
    {
      v38 = sub_AF7970(&v38, &v39);
    }
    v39 = v49;
    if ( v49 && *(_BYTE *)v49 == 1 )
    {
      v21 = *(_QWORD *)(v49 + 136);
      v22 = *(__int64 **)(v21 + 24);
      v23 = *(_DWORD *)(v21 + 32);
      if ( v23 > 0x40 )
      {
        v24 = *v22;
      }
      else
      {
        v24 = 0;
        if ( v23 )
          v24 = (__int64)((_QWORD)v22 << (64 - (unsigned __int8)v23)) >> (64 - (unsigned __int8)v23);
      }
      v40 = v24;
      v38 = sub_AF6E60(&v38, &v40);
    }
    else
    {
      v38 = sub_AF7970(&v38, &v39);
    }
    v39 = v50;
    if ( v50 && *(_BYTE *)v50 == 1 )
    {
      v25 = *(_QWORD *)(v50 + 136);
      v26 = *(__int64 **)(v25 + 24);
      v27 = *(_DWORD *)(v25 + 32);
      if ( v27 > 0x40 )
      {
        v28 = *v26;
      }
      else
      {
        v28 = 0;
        if ( v27 )
          v28 = (__int64)((_QWORD)v26 << (64 - (unsigned __int8)v27)) >> (64 - (unsigned __int8)v27);
      }
      v40 = v28;
      v38 = sub_AF6E60(&v38, &v40);
    }
    else
    {
      v38 = sub_AF7970(&v38, &v39);
    }
    v29 = v4 - 1;
    v30 = sub_AF95C0(&v38, &v41, &v42, &v43, v44, v47, &v46);
    v31 = *a2;
    v32 = v29 & v30;
    v33 = 0;
    v34 = 1;
    v35 = (_QWORD *)(v37 + 8LL * v32);
    v36 = *v35;
    if ( *a2 == *v35 )
    {
LABEL_44:
      *a3 = v35;
      return 1;
    }
    else
    {
      while ( v36 != -4096 )
      {
        if ( v36 != -8192 || v33 )
          v35 = v33;
        v32 = v29 & (v34 + v32);
        v36 = *(_QWORD *)(v37 + 8LL * v32);
        if ( v36 == v31 )
        {
          v35 = (_QWORD *)(v37 + 8LL * v32);
          goto LABEL_44;
        }
        ++v34;
        v33 = v35;
        v35 = (_QWORD *)(v37 + 8LL * v32);
      }
      if ( !v33 )
        v33 = v35;
      *a3 = v33;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
