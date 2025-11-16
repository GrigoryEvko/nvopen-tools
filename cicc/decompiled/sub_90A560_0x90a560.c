// Function: sub_90A560
// Address: 0x90a560
//
__int64 __fastcall sub_90A560(__int64 *a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  size_t v3; // r14
  void *v4; // r13
  __int64 v5; // rdx
  int v6; // r14d
  unsigned int v7; // ebx
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r14
  __int64 v12; // r12
  __int64 v13; // [rsp-88h] [rbp-88h]
  __int64 v14; // [rsp-80h] [rbp-80h]
  const char *v15; // [rsp-68h] [rbp-68h] BYREF
  char v16; // [rsp-48h] [rbp-48h]
  char v17; // [rsp-47h] [rbp-47h]

  result = a1[52];
  v2 = a1[51];
  if ( v2 != result )
  {
    result -= v2;
    if ( 0xAAAAAAAAAAAAAAABLL * (result >> 3) )
    {
      if ( result < 0 )
        sub_4262D8((__int64)"vector::_M_default_append");
      v3 = 0x5555555555555558LL * (result >> 3);
      v14 = v3;
      v4 = (void *)sub_22077B0(v3);
      memset(v4, 0, v3);
      v5 = a1[51];
      v6 = -1431655765 * ((a1[52] - v5) >> 3);
      if ( v6 )
      {
        v7 = 0;
        while ( 1 )
        {
          v8 = v7++;
          *((_QWORD *)v4 + v8) = sub_ADAFB0(*(_QWORD *)(v5 + 24 * v8 + 16), a1[87]);
          if ( v6 == v7 )
            break;
          v5 = a1[51];
        }
      }
      v9 = sub_BCD420(a1[87], v14 >> 3);
      v10 = *a1;
      v11 = v9;
      v17 = 1;
      v13 = sub_AD1300(v9, v4, v14 >> 3);
      v15 = "llvm.used";
      v16 = 3;
      v12 = sub_BD2C40(88, unk_3F0FAE8);
      if ( v12 )
        ((void (__fastcall *)(__int64, __int64, __int64, _QWORD, __int64, __int64, const char **, _QWORD, _QWORD))sub_B30000)(
          v12,
          v10,
          v11,
          0,
          6,
          v13,
          &v15,
          0,
          0);
      sub_B31A00(v12, "llvm.metadata", 13);
      return j_j___libc_free_0(v4, v14);
    }
  }
  return result;
}
