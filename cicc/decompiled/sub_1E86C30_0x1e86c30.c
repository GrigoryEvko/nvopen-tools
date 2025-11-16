// Function: sub_1E86C30
// Address: 0x1e86c30
//
__int64 __fastcall sub_1E86C30(__int64 a1, const char *a2, unsigned __int64 a3)
{
  void *v5; // rax
  __int64 v6; // rax
  int v7; // edx
  __int64 v8; // rsi
  int v9; // edx
  int v10; // edi
  unsigned int v11; // eax
  __int64 v12; // rcx
  void *v13; // r13
  void *v14; // rax
  __int64 v16; // [rsp-10h] [rbp-40h]
  __int64 v17[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_1E869F0(a1, a2, *(_QWORD *)(a3 + 24));
  v5 = sub_16E8CB0();
  sub_1263B40((__int64)v5, "- instruction: ");
  v6 = *(_QWORD *)(a1 + 584);
  if ( v6 )
  {
    v7 = *(_DWORD *)(v6 + 384);
    if ( v7 )
    {
      v8 = *(_QWORD *)(v6 + 368);
      v9 = v7 - 1;
      v10 = 1;
      v11 = v9 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v12 = *(_QWORD *)(v8 + 16LL * v11);
      if ( a3 == v12 )
      {
LABEL_4:
        v13 = sub_16E8CB0();
        v17[0] = sub_1E85F30(*(_QWORD *)(a1 + 584), a3);
        sub_1F10810(v17, v13);
        sub_1549FC0((__int64)v13, 9u);
      }
      else
      {
        while ( v12 != -8 )
        {
          v11 = v9 & (v10 + v11);
          v12 = *(_QWORD *)(v8 + 16LL * v11);
          if ( a3 == v12 )
            goto LABEL_4;
          ++v10;
        }
      }
    }
  }
  v14 = sub_16E8CB0();
  sub_1E1A330(a3, (__int64)v14, 1, 0, 0, 1, 0);
  return v16;
}
