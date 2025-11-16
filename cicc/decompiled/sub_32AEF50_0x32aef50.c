// Function: sub_32AEF50
// Address: 0x32aef50
//
__int64 __fastcall sub_32AEF50(_QWORD *a1, unsigned int a2)
{
  __int64 v2; // r12
  unsigned __int64 *v4; // rax
  unsigned int v5; // r13d
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rsi
  unsigned __int64 v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rax
  unsigned __int64 v14; // rsi

  v2 = a2;
  v4 = (unsigned __int64 *)a1[18];
  v5 = *((_DWORD *)a1 + 35);
  v6 = *v4;
  if ( *v4 )
  {
    *v4 = *(_QWORD *)v6;
LABEL_3:
    memset((void *)v6, 0, 0xC0u);
    v7 = v6 & 0xFFFFFFFFFFFFFFC0LL;
    goto LABEL_4;
  }
  v14 = v4[1];
  v4[11] += 192LL;
  v7 = (v14 + 63) & 0xFFFFFFFFFFFFFFC0LL;
  if ( v4[2] < v7 + 192 || !v14 )
  {
    v6 = sub_9D1E70((__int64)(v4 + 1), 192, 192, 6);
    goto LABEL_3;
  }
  v4[1] = v7 + 192;
  if ( v7 )
  {
    v6 = (v14 + 63) & 0xFFFFFFFFFFFFFFC0LL;
    goto LABEL_3;
  }
LABEL_4:
  v8 = 0;
  if ( v5 )
  {
    do
    {
      *(_QWORD *)(v6 + v8 * 8) = a1[v8];
      *(_QWORD *)(v6 + v8 * 8 + 8) = a1[v8 + 1];
      v8 += 2;
    }
    while ( v8 != 2LL * v5 );
  }
  v9 = v5 - 1;
  *((_DWORD *)a1 + 34) = 1;
  v10 = v9 | v7;
  memset(a1, 0, 0x88u);
  v11 = *(_QWORD *)((v10 & 0xFFFFFFFFFFFFFFC0LL) + 16 * v9 + 8);
  a1[1] = v10;
  a1[9] = v11;
  v12 = *(_QWORD *)(v10 & 0xFFFFFFFFFFFFFFC0LL);
  *((_DWORD *)a1 + 35) = 1;
  *a1 = v12;
  return v2 << 32;
}
