// Function: sub_32089B0
// Address: 0x32089b0
//
void __fastcall sub_32089B0(__int64 a1)
{
  __int64 *v1; // r12
  _BYTE *v2; // rcx
  __int64 v3; // r9
  char *v4; // rdx
  char *v5; // r8
  unsigned __int8 v6; // al
  unsigned __int8 **v7; // rdx
  __int64 v8; // rbx
  _BYTE *v9; // rdx
  __int64 v10; // r15
  _BYTE *v11; // rax
  _BYTE *v12; // rsi
  unsigned __int8 v13; // al
  _BYTE *v14; // rdx
  char v15; // dl
  unsigned int v16; // eax
  __int64 *v17; // rcx
  unsigned __int8 v18; // al
  __int64 v19; // rax
  __int64 *v20; // rsi
  unsigned int v21; // eax
  unsigned int v22; // [rsp+Ch] [rbp-84h]
  __int64 *v23; // [rsp+10h] [rbp-80h]
  __int64 *v24; // [rsp+10h] [rbp-80h]
  char v25; // [rsp+18h] [rbp-78h]
  unsigned int v26; // [rsp+18h] [rbp-78h]
  char v27; // [rsp+18h] [rbp-78h]
  __int64 *v28; // [rsp+20h] [rbp-70h]
  _BYTE *v29; // [rsp+28h] [rbp-68h]
  __int64 *v30; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v31; // [rsp+38h] [rbp-58h]
  char v32; // [rsp+3Ch] [rbp-54h]
  __int64 *v33; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v34; // [rsp+48h] [rbp-48h]
  __int64 v35; // [rsp+50h] [rbp-40h] BYREF

  v1 = *(__int64 **)(a1 + 968);
  v28 = &v1[*(unsigned int *)(a1 + 976)];
  if ( v1 != v28 )
  {
    while ( 1 )
    {
      v8 = *v1;
      v29 = (_BYTE *)(*v1 - 16);
      v9 = (*v29 & 2) != 0 ? *(_BYTE **)(v8 - 32) : &v29[-8 * ((*v29 >> 2) & 0xF)];
      v10 = *((_QWORD *)v9 + 1);
      v30 = 0;
      v31 = 1;
      v32 = 0;
      v11 = (_BYTE *)sub_AF2DC0(v8);
      v12 = v11;
      if ( v11 )
      {
        if ( *v11 == 17 )
          break;
      }
      v19 = sub_AF2DC0(v8);
      if ( !v19 || *(_BYTE *)v19 != 18 )
        BUG();
      v20 = (__int64 *)(v19 + 24);
      if ( *(void **)(v19 + 24) == sub_C33340() )
        sub_C3E660((__int64)&v33, (__int64)v20);
      else
        sub_C3A850((__int64)&v33, v20);
      v21 = v34;
      v34 = 0;
      if ( v31 <= 0x40 || !v30 )
      {
        v30 = v33;
        v31 = v21;
        v32 = 1;
        goto LABEL_26;
      }
      v24 = v33;
      v26 = v21;
      j_j___libc_free_0_0((unsigned __int64)v30);
      v32 = 1;
      v30 = v24;
      v31 = v26;
      if ( v34 > 0x40 )
        goto LABEL_24;
LABEL_26:
      v18 = *(_BYTE *)(v8 - 16);
      if ( (v18 & 2) != 0 )
      {
        v2 = *(_BYTE **)(*(_QWORD *)(v8 - 32) + 16LL);
        if ( !v2 )
          goto LABEL_28;
      }
      else
      {
        v2 = *(_BYTE **)&v29[-8 * ((v18 >> 2) & 0xF) + 16];
        if ( !v2 )
        {
LABEL_28:
          v5 = 0;
          goto LABEL_5;
        }
      }
      v2 = (_BYTE *)sub_B91420((__int64)v2);
      v5 = v4;
LABEL_5:
      sub_3205680((__int64)&v33, a1, v10, v2, v5, v3);
      v6 = *(_BYTE *)(v8 - 16);
      if ( (v6 & 2) != 0 )
        v7 = *(unsigned __int8 ***)(v8 - 32);
      else
        v7 = (unsigned __int8 **)&v29[-8 * ((v6 >> 2) & 0xF)];
      sub_32086D0(a1, v7[3], (__int64)&v30, (__int64)&v33, (__int64)&v30);
      if ( v33 != &v35 )
        j_j___libc_free_0((unsigned __int64)v33);
      if ( v31 > 0x40 )
      {
        if ( v30 )
          j_j___libc_free_0_0((unsigned __int64)v30);
      }
      if ( v28 == ++v1 )
        return;
    }
    v13 = *(_BYTE *)(v8 - 16);
    if ( (v13 & 2) != 0 )
      v14 = *(_BYTE **)(v8 - 32);
    else
      v14 = &v29[-8 * ((v13 >> 2) & 0xF)];
    v15 = sub_32120E0(*((_QWORD *)v14 + 3));
    v16 = *((_DWORD *)v12 + 8);
    v34 = v16;
    if ( v16 > 0x40 )
    {
      v27 = v15;
      sub_C43780((__int64)&v33, (const void **)v12 + 3);
      v16 = v34;
      v17 = v33;
      v15 = v27;
    }
    else
    {
      v17 = (__int64 *)*((_QWORD *)v12 + 3);
      v33 = v17;
    }
    v34 = 0;
    if ( v31 <= 0x40 || !v30 )
    {
      v30 = v17;
      v31 = v16;
      v32 = v15;
      goto LABEL_26;
    }
    v22 = v16;
    v23 = v17;
    v25 = v15;
    j_j___libc_free_0_0((unsigned __int64)v30);
    v30 = v23;
    v31 = v22;
    v32 = v25;
    if ( v34 <= 0x40 )
      goto LABEL_26;
LABEL_24:
    if ( v33 )
      j_j___libc_free_0_0((unsigned __int64)v33);
    goto LABEL_26;
  }
}
