// Function: sub_31F93A0
// Address: 0x31f93a0
//
__int64 __fastcall sub_31F93A0(__int64 a1, unsigned __int16 a2)
{
  __int64 v3; // rdi
  void (*v4)(); // rax
  __int64 *v5; // rdi
  __int64 v6; // rax
  __int64 (*v7)(); // rdx
  __int64 v9; // r12
  void (*v10)(); // r15
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  const char *v14; // rdx
  __int64 v15; // rax
  _QWORD v16[4]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v17; // [rsp+20h] [rbp-40h]

  v3 = *(_QWORD *)(a1 + 528);
  v4 = *(void (**)())(*(_QWORD *)v3 + 120LL);
  v16[0] = "Record length";
  v17 = 259;
  if ( v4 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v4)(v3, v16, 1);
    v3 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v3 + 536LL))(v3, 2, 2);
  v5 = *(__int64 **)(a1 + 528);
  v6 = *v5;
  v7 = *(__int64 (**)())(*v5 + 96);
  if ( v7 != sub_C13EE0 )
  {
    if ( ((unsigned __int8 (__fastcall *)(__int64 *))v7)(v5) )
    {
      v9 = *(_QWORD *)(a1 + 528);
      v10 = *(void (**)())(*(_QWORD *)v9 + 120LL);
      v11 = sub_37079F0(v5);
      v13 = v11 + 40 * v12;
      if ( v11 == v13 )
      {
LABEL_14:
        v15 = 0;
        v14 = byte_3F871B3;
      }
      else
      {
        while ( a2 != *(_WORD *)(v11 + 32) )
        {
          v11 += 40;
          if ( v13 == v11 )
            goto LABEL_14;
        }
        v14 = *(const char **)v11;
        v15 = *(_QWORD *)(v11 + 8);
      }
      v16[2] = v14;
      v17 = 1283;
      v16[0] = "Record kind: ";
      v16[3] = v15;
      if ( v10 != nullsub_98 )
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v10)(v9, v16, 1);
    }
    v5 = *(__int64 **)(a1 + 528);
    v6 = *v5;
  }
  return (*(__int64 (__fastcall **)(__int64 *, _QWORD, __int64))(v6 + 536))(v5, a2, 2);
}
