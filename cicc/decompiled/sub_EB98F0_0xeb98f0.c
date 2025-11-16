// Function: sub_EB98F0
// Address: 0xeb98f0
//
__int64 __fastcall sub_EB98F0(__int64 a1, char a2)
{
  unsigned int v4; // r12d
  _BOOL8 v5; // rsi
  unsigned __int8 v7; // al
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 *v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // [rsp+8h] [rbp-98h] BYREF
  const char *v14; // [rsp+10h] [rbp-90h] BYREF
  const char *v15; // [rsp+18h] [rbp-88h]
  const char *v16; // [rsp+20h] [rbp-80h] BYREF
  char v17; // [rsp+40h] [rbp-60h]
  char v18; // [rsp+41h] [rbp-5Fh]
  const char *v19[4]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v20; // [rsp+70h] [rbp-30h]

  v13 = 0;
  v4 = sub_EAC8B0(a1, &v13);
  if ( !(_BYTE)v4 && v13 != 255 )
  {
    v18 = 1;
    v5 = 1;
    v14 = 0;
    v15 = 0;
    v16 = "unsupported encoding.";
    v17 = 3;
    if ( (v13 & 0xFFFFFFFFFFFFFF00LL) == 0 && ((unsigned int)(v13 & 7) - 2 <= 2 || (v13 & 7) == 0) )
      v5 = (v13 & 0x60) != 0;
    if ( (unsigned __int8)sub_ECE0A0(a1, v5, &v16) )
      return 1;
    v20 = 259;
    v19[0] = "expected comma";
    if ( (unsigned __int8)sub_ECE210(a1, 26, v19) )
      return 1;
    v19[0] = "expected identifier in directive";
    v20 = 259;
    v7 = sub_EB61F0(a1, (__int64 *)&v14);
    if ( (unsigned __int8)sub_ECE0A0(a1, v7, v19) )
      return 1;
    v4 = sub_ECE000(a1);
    if ( (_BYTE)v4 )
    {
      return 1;
    }
    else
    {
      v8 = *(_QWORD *)(a1 + 224);
      v20 = 261;
      v19[0] = v14;
      v19[1] = v15;
      v9 = sub_E6C460(v8, v19);
      v10 = *(__int64 **)(a1 + 232);
      v11 = v9;
      v12 = *v10;
      if ( a2 )
        (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(v12 + 904))(v10, v11, (unsigned int)v13);
      else
        (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(v12 + 912))(v10, v11, (unsigned int)v13);
    }
  }
  return v4;
}
