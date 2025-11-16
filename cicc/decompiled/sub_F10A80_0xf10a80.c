// Function: sub_F10A80
// Address: 0xf10a80
//
__int64 __fastcall sub_F10A80(unsigned __int64 *a1, __int64 *a2)
{
  __int64 *v2; // r13
  __int64 v3; // rdx
  __int64 v5; // r14
  __int64 result; // rax
  __int64 v7; // r14
  unsigned __int64 *v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  __int16 v13; // dx
  __int16 v14; // cx
  char v15; // al
  char v16; // cl
  __int64 v17; // rax
  unsigned __int8 v18; // [rsp+Fh] [rbp-41h] BYREF
  unsigned __int64 *v19; // [rsp+10h] [rbp-40h] BYREF
  __int64 *v20; // [rsp+18h] [rbp-38h]
  unsigned __int8 *v21; // [rsp+20h] [rbp-30h]

  v2 = (__int64 *)*(a2 - 4);
  if ( *(_BYTE *)v2 <= 0x15u )
    return 0;
  v3 = v2[2];
  if ( v3 )
  {
    if ( !*(_QWORD *)(v3 + 8) )
      return 0;
  }
  v5 = 0;
  if ( *(_BYTE *)v2 == 22 )
  {
    v11 = *(_QWORD *)(sub_B43CB0((__int64)a2) + 80);
    if ( v11 )
      v11 -= 24;
    v12 = sub_AA5BA0(v11);
    v14 = v13;
    v8 = (unsigned __int64 *)v12;
    if ( v12 )
    {
      v15 = v14;
      v16 = HIBYTE(v14);
    }
    else
    {
      v16 = 0;
      v15 = 0;
    }
    LOBYTE(v5) = v15;
    v17 = v5;
    BYTE1(v17) = v16;
    v7 = v17;
  }
  else
  {
    sub_B445D0((__int64)&v19, (char *)*(a2 - 4));
    result = (unsigned __int8)v21;
    if ( !(_BYTE)v21 )
      return result;
    v7 = (unsigned __int16)v20;
    v8 = v19;
  }
  if ( !v8 )
    BUG();
  v9 = (__int64)(v8 - 3);
  if ( *((_BYTE *)v8 - 24) == 85
    && (v10 = *(v8 - 7)) != 0
    && !*(_BYTE *)v10
    && *(_QWORD *)(v10 + 24) == v8[7]
    && (*(_BYTE *)(v10 + 33) & 0x20) != 0
    && (unsigned int)(*(_DWORD *)(v10 + 36) - 68) <= 3 )
  {
    LOWORD(v7) = 0;
    v18 = 0;
    v9 = sub_B46B10(v9, 0);
    v8 = (unsigned __int64 *)(v9 + 24);
  }
  else
  {
    v18 = 0;
    LOBYTE(v7) = 0;
  }
  if ( a2 != (__int64 *)v9 )
  {
    sub_B44550(a2, *(_QWORD *)(v9 + 40), v8, v7);
    v18 = 1;
  }
  v19 = a1;
  v20 = a2;
  v21 = &v18;
  sub_BD79D0(v2, a2, (unsigned __int8 (__fastcall *)(__int64, __int64 *))sub_F06520, (__int64)&v19);
  return v18;
}
