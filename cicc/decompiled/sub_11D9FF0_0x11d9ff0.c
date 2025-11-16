// Function: sub_11D9FF0
// Address: 0x11d9ff0
//
__int64 __fastcall sub_11D9FF0(__int64 a1, __int16 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rsi
  __int64 v11; // rdi
  __int64 v12; // rdx
  int v13; // ecx
  __int64 *v14; // rax
  __int64 v15; // r8
  const char *v16; // rax
  __int64 result; // rax
  bool v18; // zf
  __int64 *v19; // rax
  int v20; // [rsp+Ch] [rbp-94h] BYREF
  _QWORD v21[4]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v22; // [rsp+30h] [rbp-70h]
  unsigned __int64 v23[2]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v24; // [rsp+50h] [rbp-50h]
  __int64 v25; // [rsp+58h] [rbp-48h]
  __int16 v26; // [rsp+60h] [rbp-40h]

  v7 = (_QWORD *)a6;
  v11 = a5;
  v20 = 0;
  if ( !sub_C93C90(a5, a6, 0, v23) && (v12 = LODWORD(v23[0]), v13 = v23[0], v23[0] == LODWORD(v23[0])) )
  {
    v20 = v23[0];
    if ( LODWORD(v23[0]) <= 0xFF )
      goto LABEL_7;
    v19 = sub_CEADF0();
    v24 = a5;
    v15 = (__int64)v19;
    v26 = 1283;
    v23[0] = (unsigned __int64)"'";
    v16 = "' value must be in the range [0, 255]!";
    v25 = a6;
    v21[0] = v23;
  }
  else
  {
    v14 = sub_CEADF0();
    v24 = a5;
    v26 = 1283;
    v15 = (__int64)v14;
    v25 = a6;
    v21[0] = v23;
    v23[0] = (unsigned __int64)"'";
    v16 = "' value invalid for uint argument!";
  }
  v21[2] = v16;
  v7 = v21;
  v11 = a1;
  v22 = 770;
  result = sub_C53280(a1, (__int64)v21, 0, 0, v15);
  if ( (_BYTE)result )
    return result;
  v13 = v20;
LABEL_7:
  v18 = *(_QWORD *)(a1 + 184) == 0;
  *(_DWORD *)(a1 + 136) = v13;
  *(_WORD *)(a1 + 14) = a2;
  if ( v18 )
    sub_4263D6(v11, v7, v12);
  (*(void (__fastcall **)(__int64, int *))(a1 + 192))(a1 + 168, &v20);
  return 0;
}
