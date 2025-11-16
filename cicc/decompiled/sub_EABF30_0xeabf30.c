// Function: sub_EABF30
// Address: 0xeabf30
//
__int64 __fastcall sub_EABF30(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rdx
  __int64 v4; // r13
  __int64 result; // rax
  _QWORD v6[4]; // [rsp+0h] [rbp-80h] BYREF
  __int16 v7; // [rsp+20h] [rbp-60h]
  _QWORD v8[4]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v9; // [rsp+50h] [rbp-30h]

  v2 = sub_EABDC0(a1);
  v4 = v3;
  result = sub_ECE000(a1);
  if ( !(_BYTE)result )
  {
    if ( v4 )
    {
      v6[2] = v2;
      v7 = 1283;
      v6[0] = ".abort '";
      v8[0] = v6;
      v6[3] = v4;
      v8[2] = "' detected. Assembly stopping";
      v9 = 770;
    }
    else
    {
      v8[0] = ".abort detected. Assembly stopping";
      v9 = 259;
    }
    return sub_ECDA70(a1, a2, v8, 0, 0);
  }
  return result;
}
