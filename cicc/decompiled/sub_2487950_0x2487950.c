// Function: sub_2487950
// Address: 0x2487950
//
unsigned __int64 __fastcall sub_2487950(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // r14
  __int64 v3; // rbx
  _QWORD *v4; // r12
  unsigned __int64 result; // rax
  char v6; // al
  char *v7; // rax
  size_t v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // [rsp+8h] [rbp-58h]
  _QWORD v14[4]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v15; // [rsp+30h] [rbp-30h]

  v1 = sub_AC9B20(*(_QWORD *)a1, (char *)qword_4FE91E8, qword_4FE91F0, 1);
  BYTE4(v13) = 0;
  v2 = *(_QWORD **)(v1 + 8);
  v3 = v1;
  v14[1] = 29;
  v15 = 261;
  v14[0] = "__memprof_default_options_str";
  v4 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v4 )
    sub_B30000((__int64)v4, a1, v2, 1, 4, v3, (__int64)v14, 0, 0, v13, 0);
  result = *(unsigned int *)(a1 + 284);
  if ( (unsigned int)result > 8 || (v12 = 292, !_bittest64(&v12, result)) )
  {
    v6 = *((_BYTE *)v4 + 32);
    *((_BYTE *)v4 + 32) = v6 & 0xF0;
    if ( (v6 & 0x30) != 0 )
      *((_BYTE *)v4 + 33) |= 0x40u;
    v7 = (char *)sub_BD5D20((__int64)v4);
    v9 = sub_BAA410(a1, v7, v8);
    return (unsigned __int64)sub_B2F990((__int64)v4, v9, v10, v11);
  }
  return result;
}
