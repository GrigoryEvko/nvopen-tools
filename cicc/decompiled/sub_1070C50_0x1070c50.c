// Function: sub_1070C50
// Address: 0x1070c50
//
__int64 __fastcall sub_1070C50(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r9
  __int64 v7; // rcx
  int v8; // eax
  int v9; // edi
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // r10
  __int64 v13; // rbx
  int v14; // eax
  int v15; // r11d

  if ( (a2[9] & 0x70) == 0x20 )
    return sub_1070D00(a1, a2, a3, a4, a3);
  v6 = *(_QWORD *)(a1 + 232);
  v7 = *(_QWORD *)(*(_QWORD *)a2 + 8LL);
  v8 = *(_DWORD *)(a1 + 248);
  if ( !v8 )
    goto LABEL_7;
  v9 = v8 - 1;
  v10 = (v8 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v11 = (__int64 *)(v6 + 16LL * v10);
  v12 = *v11;
  if ( v7 != *v11 )
  {
    v14 = 1;
    while ( v12 != -4096 )
    {
      v15 = v14 + 1;
      v10 = v9 & (v14 + v10);
      v11 = (__int64 *)(v6 + 16LL * v10);
      v12 = *v11;
      if ( v7 == *v11 )
        goto LABEL_5;
      v14 = v15;
    }
LABEL_7:
    v13 = 0;
    return v13 + sub_E5C4C0(a3, (__int64)a2);
  }
LABEL_5:
  v13 = v11[1];
  return v13 + sub_E5C4C0(a3, (__int64)a2);
}
