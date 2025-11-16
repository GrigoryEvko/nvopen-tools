// Function: sub_28AA9B0
// Address: 0x28aa9b0
//
_BYTE *__fastcall sub_28AA9B0(__int64 *a1)
{
  __int64 v1; // rbx
  _QWORD *v2; // rdi
  __int64 v3; // rcx
  __int64 v4; // rax
  __int64 v5; // rsi
  int v6; // eax
  int v7; // r9d
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // r10
  __int64 v11; // rsi
  unsigned __int8 *v12; // rax
  int v13; // ecx
  unsigned __int8 *v14; // rdx
  _BYTE *result; // rax
  int v16; // eax
  int v17; // r11d

  v1 = *a1;
  v2 = sub_103E0E0(*(_QWORD **)(*(_QWORD *)*a1 + 40LL));
  v3 = **(_QWORD **)(v1 + 8);
  v4 = v2[1];
  v5 = *(_QWORD *)(v4 + 40);
  v6 = *(_DWORD *)(v4 + 56);
  if ( !v6 )
  {
LABEL_11:
    v11 = 0;
    goto LABEL_4;
  }
  v7 = v6 - 1;
  v8 = (v6 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v9 = (__int64 *)(v5 + 16LL * v8);
  v10 = *v9;
  if ( v3 != *v9 )
  {
    v16 = 1;
    while ( v10 != -4096 )
    {
      v17 = v16 + 1;
      v8 = v7 & (v16 + v8);
      v9 = (__int64 *)(v5 + 16LL * v8);
      v10 = *v9;
      if ( v3 == *v9 )
        goto LABEL_3;
      v16 = v17;
    }
    goto LABEL_11;
  }
LABEL_3:
  v11 = v9[1];
LABEL_4:
  v12 = (unsigned __int8 *)(*(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD))(*v2 + 16LL))(
                             v2,
                             v11,
                             *(_QWORD *)(v1 + 16));
  v13 = *v12;
  v14 = v12;
  result = 0;
  if ( (unsigned int)(v13 - 26) <= 1 )
  {
    result = (_BYTE *)*((_QWORD *)v14 + 9);
    if ( result )
    {
      if ( *result != 85 )
        return 0;
    }
  }
  return result;
}
