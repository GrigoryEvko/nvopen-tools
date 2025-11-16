// Function: sub_18E67F0
// Address: 0x18e67f0
//
void __fastcall sub_18E67F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // r12
  int v8; // eax
  __int64 v9; // rdi
  int v10; // eax
  __int64 v11; // rdx
  unsigned __int64 *v12; // rsi
  char *v13; // rdi
  char *v14[2]; // [rsp+0h] [rbp-C0h] BYREF
  _BYTE v15[128]; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v16; // [rsp+90h] [rbp-30h]
  int v17; // [rsp+98h] [rbp-28h]

  v6 = a1;
  v14[1] = (char *)0x800000000LL;
  v8 = *(_DWORD *)(a1 + 8);
  v14[0] = v15;
  if ( v8 )
    sub_18E63F0((__int64)v14, (char **)a1, a3, a4, a5, a6);
  v9 = *(_QWORD *)(a1 + 144);
  v10 = *(_DWORD *)(a1 + 152);
  v16 = *(_QWORD *)(a1 + 144);
  v17 = v10;
  while ( 1 )
  {
    v12 = *(unsigned __int64 **)(v6 - 16);
    if ( *(_QWORD *)v9 == *v12 )
      break;
    v11 = *(_DWORD *)(*(_QWORD *)v9 + 8LL) >> 8;
    if ( (unsigned int)v11 >= *(_DWORD *)(*v12 + 8) >> 8 )
      goto LABEL_8;
LABEL_5:
    sub_18E63F0(v6, (char **)(v6 - 160), v11, a4, a5, a6);
    v9 = v16;
    *(_QWORD *)(v6 + 144) = *(_QWORD *)(v6 - 16);
    *(_DWORD *)(v6 + 152) = *(_DWORD *)(v6 - 8);
    v6 -= 160;
  }
  if ( (int)sub_16A9900(v9 + 24, v12 + 3) < 0 )
    goto LABEL_5;
LABEL_8:
  sub_18E63F0(v6, v14, v11, a4, a5, a6);
  v13 = v14[0];
  *(_QWORD *)(v6 + 144) = v16;
  *(_DWORD *)(v6 + 152) = v17;
  if ( v13 != v15 )
    _libc_free((unsigned __int64)v13);
}
