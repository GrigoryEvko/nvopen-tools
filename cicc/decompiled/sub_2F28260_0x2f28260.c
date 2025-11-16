// Function: sub_2F28260
// Address: 0x2f28260
//
__int64 __fastcall sub_2F28260(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 (*v10)(); // rdx
  __int64 v11; // rax
  __int64 v12; // rbx
  _QWORD *v13; // r13
  char v14; // r12
  char v15; // al
  __int64 **v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  void **v20; // rax
  __int64 **v22; // rsi
  __int64 v23; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v24; // [rsp+8h] [rbp-A8h]
  char v25; // [rsp+10h] [rbp-A0h]
  __int64 v26; // [rsp+18h] [rbp-98h]
  __int64 v27; // [rsp+20h] [rbp-90h] BYREF
  void **v28; // [rsp+28h] [rbp-88h]
  unsigned int v29; // [rsp+30h] [rbp-80h]
  unsigned int v30; // [rsp+34h] [rbp-7Ch]
  char v31; // [rsp+3Ch] [rbp-74h]
  _BYTE v32[16]; // [rsp+40h] [rbp-70h] BYREF
  _BYTE v33[8]; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v34; // [rsp+58h] [rbp-58h]
  int v35; // [rsp+64h] [rbp-4Ch]
  int v36; // [rsp+68h] [rbp-48h]
  char v37; // [rsp+6Ch] [rbp-44h]
  _BYTE v38[64]; // [rsp+70h] [rbp-40h] BYREF

  v8 = a3[4];
  v9 = a3[2];
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v23 = v8;
  v10 = *(__int64 (**)())(*(_QWORD *)v9 + 128LL);
  v11 = 0;
  if ( v10 != sub_2DAC790 )
    v11 = ((__int64 (__fastcall *)(__int64, __int64, __int64 (*)(), __int64))v10)(v9, a2, v10, a4);
  v12 = a3[41];
  v13 = a3 + 40;
  v14 = 0;
  v24 = v11;
  if ( (_QWORD *)v12 == v13 )
    goto LABEL_16;
  do
  {
    v15 = sub_2F26B60(&v23, v12, (__int64)v10, a4, a5, a6);
    v12 = *(_QWORD *)(v12 + 8);
    v14 |= v15;
  }
  while ( v13 != (_QWORD *)v12 );
  if ( !v14 )
  {
LABEL_16:
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  sub_2EAFFB0((__int64)&v27);
  if ( v35 == v36 )
  {
    if ( v31 )
    {
      v20 = v28;
      v22 = (__int64 **)&v28[v30];
      v17 = v30;
      v16 = (__int64 **)v28;
      if ( v28 != (void **)v22 )
      {
        while ( *v16 != &qword_4F82400 )
        {
          if ( v22 == ++v16 )
          {
LABEL_11:
            while ( *v20 != &unk_4F82408 )
            {
              if ( ++v20 == (void **)v16 )
                goto LABEL_24;
            }
            goto LABEL_12;
          }
        }
        goto LABEL_12;
      }
      goto LABEL_24;
    }
    if ( sub_C8CA60((__int64)&v27, (__int64)&qword_4F82400) )
      goto LABEL_12;
  }
  if ( !v31 )
  {
LABEL_26:
    sub_C8CC70((__int64)&v27, (__int64)&unk_4F82408, (__int64)v16, v17, v18, v19);
    goto LABEL_12;
  }
  v20 = v28;
  v17 = v30;
  v16 = (__int64 **)&v28[v30];
  if ( v16 != (__int64 **)v28 )
    goto LABEL_11;
LABEL_24:
  if ( (unsigned int)v17 >= v29 )
    goto LABEL_26;
  v30 = v17 + 1;
  *v16 = (__int64 *)&unk_4F82408;
  ++v27;
LABEL_12:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v32, (__int64)&v27);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v38, (__int64)v33);
  if ( !v37 )
    _libc_free(v34);
  if ( !v31 )
    _libc_free((unsigned __int64)v28);
  return a1;
}
