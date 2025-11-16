// Function: sub_114A4F0
// Address: 0x114a4f0
//
__int64 __fastcall sub_114A4F0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // r12
  __int64 *v8; // r15
  const char *v9; // rax
  __int64 v10; // rdx
  char v11; // cl
  char v12; // bl
  unsigned __int64 v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // r12
  __int64 v16; // rbx
  __int64 v17; // r13
  __int64 v18; // rdx
  unsigned int v19; // esi
  __int16 v20; // ax
  __int64 v22; // [rsp+0h] [rbp-A0h]
  char v23; // [rsp+8h] [rbp-98h]
  const char *v24; // [rsp+10h] [rbp-90h] BYREF
  __int64 v25; // [rsp+18h] [rbp-88h]
  __int64 *v26; // [rsp+20h] [rbp-80h]
  __int64 v27; // [rsp+28h] [rbp-78h]
  __int16 v28; // [rsp+30h] [rbp-70h]
  _BYTE v29[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v30; // [rsp+60h] [rbp-40h]

  v8 = *(__int64 **)(a1 + 32);
  v9 = sub_BD5D20(a2);
  v11 = *((_BYTE *)a4 + 32);
  if ( v11 )
  {
    if ( v11 == 1 )
    {
      v24 = v9;
      v25 = v10;
      v28 = 261;
    }
    else
    {
      if ( *((_BYTE *)a4 + 33) == 1 )
      {
        v4 = a4[1];
        a4 = (__int64 *)*a4;
      }
      else
      {
        v11 = 2;
      }
      v24 = v9;
      v25 = v10;
      v26 = a4;
      v27 = v4;
      LOBYTE(v28) = 5;
      HIBYTE(v28) = v11;
    }
  }
  else
  {
    v28 = 256;
  }
  v22 = *(_QWORD *)(a2 - 32);
  v12 = *(_WORD *)(a2 + 2) & 1;
  _BitScanReverse64(&v13, 1LL << (*(_WORD *)(a2 + 2) >> 1));
  v30 = 257;
  v23 = 63 - (v13 ^ 0x3F);
  v14 = sub_BD2C40(80, unk_3F10A14);
  v15 = (__int64)v14;
  if ( v14 )
    sub_B4D190((__int64)v14, a3, v22, (__int64)v29, v12, v23, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)v8[11] + 16LL))(
    v8[11],
    v15,
    &v24,
    v8[7],
    v8[8]);
  v16 = *v8;
  v17 = *v8 + 16LL * *((unsigned int *)v8 + 2);
  if ( *v8 != v17 )
  {
    do
    {
      v18 = *(_QWORD *)(v16 + 8);
      v19 = *(_DWORD *)v16;
      v16 += 16;
      sub_B99FD0(v15, v19, v18);
    }
    while ( v17 != v16 );
  }
  v20 = *(_WORD *)(a2 + 2) & 0x380;
  *(_BYTE *)(v15 + 72) = *(_BYTE *)(a2 + 72);
  *(_WORD *)(v15 + 2) = v20 | *(_WORD *)(v15 + 2) & 0xFC7F;
  sub_F57A40(v15, a2);
  return v15;
}
