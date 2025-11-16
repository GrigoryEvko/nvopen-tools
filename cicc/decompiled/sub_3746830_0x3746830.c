// Function: sub_3746830
// Address: 0x3746830
//
__int64 __fastcall sub_3746830(__int64 *a1, __int64 a2)
{
  unsigned int v4; // eax
  __int64 v5; // r15
  unsigned int v6; // r14d
  __int64 result; // rax
  __int64 v8; // rdi
  __int64 v9; // r15
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 (__fastcall *v13)(__int64, __int64, unsigned int, __int64); // rbx
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // rsi
  int v17; // ecx
  unsigned int v18; // eax
  __int64 v19; // rdx
  int v20; // r8d
  unsigned int v21; // [rsp+Ch] [rbp-54h]
  _BYTE v22[8]; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int16 v23; // [rsp+18h] [rbp-48h]

  v4 = sub_2D5BAE0(a1[16], a1[14], *(__int64 **)(a2 + 8), 1);
  if ( !(_WORD)v4 )
    return 0;
  v5 = a1[16];
  v6 = v4;
  if ( *(_QWORD *)(v5 + 8LL * (unsigned __int16)v4 + 112) )
    goto LABEL_3;
  if ( (unsigned __int16)(v4 - 5) > 1u && (_WORD)v4 != 2 )
    return 0;
  v13 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v5 + 592LL);
  v14 = sub_BD5C60(a2);
  if ( v13 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)v22, v5, v14, (unsigned __int16)v6, 0);
    v6 = v23;
  }
  else
  {
    v6 = v13(v5, v14, (unsigned __int16)v6, 0);
  }
LABEL_3:
  result = sub_3742170((__int64)a1, a2);
  if ( !(_DWORD)result )
  {
    if ( *(_BYTE *)a2 > 0x1Cu )
    {
      v8 = a1[5];
      if ( *(_BYTE *)a2 != 60 )
        return sub_374D810(v8, a2);
      v15 = *(_DWORD *)(v8 + 272);
      v16 = *(_QWORD *)(v8 + 256);
      if ( !v15 )
        return sub_374D810(v8, a2);
      v17 = v15 - 1;
      v18 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v19 = *(_QWORD *)(v16 + 16LL * v18);
      if ( a2 != v19 )
      {
        v20 = 1;
        while ( v19 != -4096 )
        {
          v18 = v17 & (v20 + v18);
          v19 = *(_QWORD *)(v16 + 16LL * v18);
          if ( a2 == v19 )
            goto LABEL_11;
          ++v20;
        }
        return sub_374D810(v8, a2);
      }
    }
LABEL_11:
    v9 = sub_3741710((__int64)a1);
    v21 = sub_3746590(a1, (unsigned __int8 *)a2, v6, v10, v11, v12);
    sub_3741740((__int64)a1, v9);
    return v21;
  }
  return result;
}
