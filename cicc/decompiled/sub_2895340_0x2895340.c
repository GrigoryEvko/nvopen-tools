// Function: sub_2895340
// Address: 0x2895340
//
char __fastcall sub_2895340(__int64 a1, int a2, int a3, __int64 *a4)
{
  __int64 v4; // rax
  bool v5; // zf
  int v8; // r14d
  int v9; // r15d
  int v10; // esi
  __int64 **v11; // rax
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v16; // [rsp+8h] [rbp-48h]
  int v17; // [rsp+1Ch] [rbp-34h]

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x1000000000LL;
  LODWORD(v4) = dword_5003CC8;
  v17 = a2;
  *(_QWORD *)(a1 + 144) = 0;
  v5 = (_DWORD)v4 == 0;
  if ( !(_DWORD)v4 )
    a2 = a3;
  LOBYTE(v4) = (_DWORD)v4 == 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_BYTE *)(a1 + 160) = v5;
  if ( a2 )
  {
    v8 = a2;
    v9 = 0;
    while ( 1 )
    {
      v10 = v17;
      if ( !(_BYTE)v4 )
        v10 = a3;
      v11 = (__int64 **)sub_BCDA70(a4, v10);
      v4 = sub_ACADE0(v11);
      v14 = *(unsigned int *)(a1 + 8);
      if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        v16 = v4;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v14 + 1, 8u, v12, v13);
        v14 = *(unsigned int *)(a1 + 8);
        v4 = v16;
      }
      ++v9;
      *(_QWORD *)(*(_QWORD *)a1 + 8 * v14) = v4;
      ++*(_DWORD *)(a1 + 8);
      if ( v8 == v9 )
        break;
      LOBYTE(v4) = *(_BYTE *)(a1 + 160);
    }
  }
  return v4;
}
