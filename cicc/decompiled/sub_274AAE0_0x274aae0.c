// Function: sub_274AAE0
// Address: 0x274aae0
//
__int64 __fastcall sub_274AAE0(__int64 a1, __m128i a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 **v24; // rdx
  __int64 v25; // r8
  __int64 v26; // r9
  void **v27; // rax
  unsigned int v28; // edi
  __int64 **v29; // rsi
  __int64 v30; // [rsp+8h] [rbp-98h]
  __int64 v31; // [rsp+10h] [rbp-90h] BYREF
  void **v32; // [rsp+18h] [rbp-88h]
  __int64 v33; // [rsp+20h] [rbp-80h]
  int v34; // [rsp+28h] [rbp-78h]
  char v35; // [rsp+2Ch] [rbp-74h]
  _BYTE v36[16]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v37; // [rsp+40h] [rbp-60h] BYREF
  _BYTE *v38; // [rsp+48h] [rbp-58h]
  __int64 v39; // [rsp+50h] [rbp-50h]
  int v40; // [rsp+58h] [rbp-48h]
  char v41; // [rsp+5Ch] [rbp-44h]
  _BYTE v42[64]; // [rsp+60h] [rbp-40h] BYREF

  v8 = sub_BC1CD0(a5, &unk_4F81450, a4);
  v30 = sub_BC1CD0(a5, &unk_4F875F0, a4);
  v9 = sub_BC1CD0(a5, &unk_4F881D0, a4);
  v10 = sub_BC1CD0(a5, &unk_4F8FAE8, a4);
  if ( !(unsigned __int8)sub_2747FC0(a4, (__int64 *)(v8 + 8), (_QWORD *)(v30 + 8), v9 + 8, v10 + 8, v11, a2) )
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v31 = 0;
  v32 = (void **)v36;
  v33 = 2;
  v34 = 0;
  v35 = 1;
  v37 = 0;
  v38 = v42;
  v39 = 2;
  v40 = 0;
  v41 = 1;
  sub_2739960((__int64)&v31, (__int64)&unk_4F81450, v12, (__int64)v36, v13, v14);
  sub_2739960((__int64)&v31, (__int64)&unk_4F875F0, v16, v17, v18, v19);
  sub_2739960((__int64)&v31, (__int64)&unk_4F881D0, v20, v21, v22, v23);
  if ( HIDWORD(v39) == v40 )
  {
    if ( v35 )
    {
      v27 = v32;
      v29 = (__int64 **)&v32[HIDWORD(v33)];
      v28 = HIDWORD(v33);
      v24 = (__int64 **)v32;
      if ( v32 != (void **)v29 )
      {
        while ( *v24 != &qword_4F82400 )
        {
          if ( v29 == ++v24 )
          {
LABEL_9:
            while ( *v27 != &unk_4F82408 )
            {
              if ( ++v27 == (void **)v24 )
                goto LABEL_14;
            }
            goto LABEL_10;
          }
        }
        goto LABEL_10;
      }
      goto LABEL_14;
    }
    if ( sub_C8CA60((__int64)&v31, (__int64)&qword_4F82400) )
      goto LABEL_10;
  }
  if ( !v35 )
  {
LABEL_16:
    sub_C8CC70((__int64)&v31, (__int64)&unk_4F82408, (__int64)v24, (__int64)v36, v25, v26);
    goto LABEL_10;
  }
  v27 = v32;
  v28 = HIDWORD(v33);
  v24 = (__int64 **)&v32[HIDWORD(v33)];
  if ( v32 != (void **)v24 )
    goto LABEL_9;
LABEL_14:
  if ( v28 >= (unsigned int)v33 )
    goto LABEL_16;
  HIDWORD(v33) = v28 + 1;
  *v24 = (__int64 *)&unk_4F82408;
  ++v31;
LABEL_10:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v36, (__int64)&v31);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v42, (__int64)&v37);
  if ( !v41 )
    _libc_free((unsigned __int64)v38);
  if ( !v35 )
    _libc_free((unsigned __int64)v32);
  return a1;
}
