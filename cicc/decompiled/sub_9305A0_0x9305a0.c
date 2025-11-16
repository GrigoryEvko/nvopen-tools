// Function: sub_9305A0
// Address: 0x9305a0
//
__int64 __fastcall sub_9305A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  const char *v6; // rdi
  __int64 v8; // rdi
  __int64 v9; // r15
  __int64 v10; // rax
  _BYTE *v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // rax
  _BYTE *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // r14
  _BYTE *v24; // [rsp-90h] [rbp-90h]
  __int64 v25; // [rsp-90h] [rbp-90h]
  int v26; // [rsp-7Ch] [rbp-7Ch] BYREF
  _BYTE *v27; // [rsp-78h] [rbp-78h] BYREF
  __int64 v28; // [rsp-70h] [rbp-70h]
  _BYTE v29[16]; // [rsp-68h] [rbp-68h] BYREF
  _QWORD *v30; // [rsp-58h] [rbp-58h]
  __int64 v31; // [rsp-50h] [rbp-50h]
  _QWORD v32[9]; // [rsp-48h] [rbp-48h] BYREF

  result = dword_4D046B4;
  if ( !dword_4D046B4 )
  {
    if ( a3 )
    {
      v6 = *(const char **)(a3 + 64);
      if ( v6 )
      {
        if ( sscanf(v6, "unroll %d", &v26) != 1 )
          sub_91B8A0("Parsing unroll count failed!", (_DWORD *)a3, 1);
        if ( v26 <= 0 )
          sub_91B8A0("Unroll count must be positive.", (_DWORD *)a3, 1);
        v8 = *(_QWORD *)(a1 + 40);
        v27 = v29;
        v28 = 0x200000000LL;
        if ( v26 == 0x7FFFFFFF )
        {
          v17 = sub_B9B140(v8, "llvm.loop.unroll.full", 21);
        }
        else
        {
          v9 = sub_B9B140(v8, "llvm.loop.unroll.count", 22);
          v24 = (_BYTE *)v26;
          v10 = sub_BCB2D0(*(_QWORD *)(a1 + 40));
          v11 = v24;
          v12 = sub_ACD640(v10, v24, 0);
          v13 = HIDWORD(v28);
          v14 = v12;
          v15 = (unsigned int)v28;
          if ( (unsigned __int64)(unsigned int)v28 + 1 > HIDWORD(v28) )
          {
            v11 = v29;
            v25 = v14;
            sub_C8D5F0(&v27, v29, (unsigned int)v28 + 1LL, 8);
            v15 = (unsigned int)v28;
            v14 = v25;
          }
          v16 = v27;
          *(_QWORD *)&v27[8 * v15] = v9;
          LODWORD(v28) = v28 + 1;
          v17 = sub_B98A20(v14, v11, v16, v13);
        }
        v18 = v17;
        v19 = (unsigned int)v28;
        if ( (unsigned __int64)(unsigned int)v28 + 1 > HIDWORD(v28) )
        {
          sub_C8D5F0(&v27, v29, (unsigned int)v28 + 1LL, 8);
          v19 = (unsigned int)v28;
        }
        *(_QWORD *)&v27[8 * v19] = v18;
        v20 = *(_QWORD *)(a1 + 40);
        LODWORD(v28) = v28 + 1;
        v21 = sub_B9C770(v20, v27, (unsigned int)v28, 0, 1);
        v22 = *(_QWORD *)(a1 + 40);
        v32[1] = v21;
        v30 = v32;
        v31 = 0x200000002LL;
        v32[0] = 0;
        v23 = sub_B9C770(v22, v32, 2, 0, 1);
        sub_BA6610(v23, 0, v23);
        result = sub_B9A090(a2, "llvm.loop", 9, v23);
        if ( v30 != v32 )
          result = _libc_free(v30, "llvm.loop");
        if ( v27 != v29 )
          return _libc_free(v27, "llvm.loop");
      }
    }
  }
  return result;
}
