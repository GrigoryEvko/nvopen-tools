// Function: sub_24876D0
// Address: 0x24876d0
//
__int64 __fastcall sub_24876D0(__int64 a1)
{
  __int64 result; // rax
  char *v2; // rax
  signed __int64 v3; // rdx
  __int64 v4; // rax
  _QWORD *v5; // r14
  __int64 v6; // rbx
  _QWORD *v7; // r13
  char v8; // al
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // [rsp+8h] [rbp-58h]
  const char *v14; // [rsp+10h] [rbp-50h] BYREF
  char v15; // [rsp+30h] [rbp-30h]
  char v16; // [rsp+31h] [rbp-2Fh]

  result = sub_BA91D0(a1, "MemProfProfileFilename", 0x16u);
  if ( result && !*(_BYTE *)result )
  {
    v2 = (char *)sub_B91420(result);
    v4 = sub_AC9B20(*(_QWORD *)a1, v2, v3, 1);
    BYTE4(v13) = 0;
    v5 = *(_QWORD **)(v4 + 8);
    v6 = v4;
    v16 = 1;
    v14 = "__memprof_profile_filename";
    v15 = 3;
    v7 = sub_BD2C40(88, unk_3F0FAE8);
    if ( v7 )
      sub_B30000((__int64)v7, a1, v5, 1, 4, v6, (__int64)&v14, 0, 0, v13, 0);
    result = *(unsigned int *)(a1 + 284);
    if ( (unsigned int)result > 8 || (v12 = 292, !_bittest64(&v12, result)) )
    {
      v8 = *((_BYTE *)v7 + 32);
      *((_BYTE *)v7 + 32) = v8 & 0xF0;
      if ( (v8 & 0x30) != 0 )
        *((_BYTE *)v7 + 33) |= 0x40u;
      v9 = sub_BAA410(a1, "__memprof_profile_filename", 0x1Au);
      return (__int64)sub_B2F990((__int64)v7, v9, v10, v11);
    }
  }
  return result;
}
