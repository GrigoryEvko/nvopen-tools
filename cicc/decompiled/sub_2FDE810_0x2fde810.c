// Function: sub_2FDE810
// Address: 0x2fde810
//
__int64 __fastcall sub_2FDE810(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx
  int v4; // eax
  unsigned __int16 *v5; // r12
  __int64 v6; // rdx
  unsigned int v7; // r9d
  __int64 v8; // r13
  unsigned __int64 v9; // rbx
  unsigned int v10; // r15d
  __int64 v11; // r8
  __int64 v12; // rdi
  char v13; // al
  int v15; // eax
  __int64 v16; // [rsp+8h] [rbp-48h]
  unsigned __int8 v17; // [rsp+17h] [rbp-39h]
  __int64 v18; // [rsp+18h] [rbp-38h]

  v3 = a3;
  v4 = *(_DWORD *)(a2 + 44);
  v5 = *(unsigned __int16 **)(a2 + 16);
  if ( (v4 & 4) == 0 && (v4 & 8) != 0 )
  {
    LOBYTE(v15) = sub_2E88A90(a2, (__int64)&dword_400000, 2);
    v3 = a3;
    LODWORD(v6) = v15;
  }
  else
  {
    v6 = (*((_QWORD *)v5 + 3) >> 22) & 1LL;
  }
  v7 = 0;
  if ( (_BYTE)v6 && (*(_DWORD *)(a2 + 40) & 0xFFFFFF) != 0 )
  {
    v8 = 0;
    v9 = 0;
    v10 = 0;
    v11 = 6LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
    do
    {
      if ( (v5[20 * *v5 + 21 + 3 * v5[8] + v9 / 2] & 2) != 0 )
      {
        v12 = v8 + *(_QWORD *)(a2 + 32);
        v13 = *(_BYTE *)v12;
        if ( *(_BYTE *)v12 )
        {
          if ( v13 == 1 || v13 == 4 )
          {
            v7 = v6;
            *(_QWORD *)(v12 + 24) = *(_QWORD *)(v3 + 40LL * v10 + 24);
          }
        }
        else
        {
          v16 = v11;
          v17 = v6;
          v18 = v3;
          sub_2EAB0C0(v12, *(_DWORD *)(v3 + 40LL * v10 + 8));
          LODWORD(v6) = v17;
          v3 = v18;
          v11 = v16;
          v7 = v17;
        }
        ++v10;
      }
      v9 += 6LL;
      v8 += 40;
    }
    while ( v11 != v9 );
  }
  return v7;
}
