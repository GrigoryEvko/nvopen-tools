// Function: sub_2B399C0
// Address: 0x2b399c0
//
_BOOL8 __fastcall sub_2B399C0(__int64 a1, char a2)
{
  bool *v2; // rax
  _DWORD *v3; // rdx
  unsigned int v4; // edx
  __int64 v6; // r12
  __int64 v7; // r14
  unsigned int v8; // r15d
  __int64 v9; // rax
  unsigned int v10; // ebx
  unsigned int v11; // eax
  int v12; // esi
  __int64 *v13; // rdi
  unsigned int v14; // eax
  __int64 v15; // r13
  unsigned int v16; // ecx
  int v17; // eax
  unsigned int v18; // ecx
  unsigned int *v19; // [rsp+8h] [rbp-48h]
  bool v20; // [rsp+17h] [rbp-39h]
  __int64 v21; // [rsp+18h] [rbp-38h]
  __int64 v22; // [rsp+18h] [rbp-38h]

  v20 = 0;
  if ( !a2 )
    v20 = sub_2B39890(*(_QWORD *)a1 + 768LL, *(_QWORD *)(a1 + 8));
  v2 = *(bool **)(a1 + 16);
  if ( !*v2 && **(_DWORD **)(a1 + 24) == **(_DWORD **)(a1 + 32) )
    *v2 = v20;
  ++**(_DWORD **)(a1 + 40);
  v3 = *(_DWORD **)(a1 + 40);
  if ( *v3 >= (unsigned int)(**(_DWORD **)(a1 + 48) + 1 - **(_DWORD **)(a1 + 32)) )
  {
    *v3 = **(_DWORD **)(a1 + 56);
    v4 = --**(_DWORD **)(a1 + 32);
    v19 = *(unsigned int **)(a1 + 32);
    if ( v4 > 1 )
    {
      v6 = *(_QWORD *)(a1 + 64);
      v7 = *(_QWORD *)(***(_QWORD ***)(v6 + 8) + 8LL);
      v8 = sub_2B1E190(*(_QWORD *)v6, v7, v4);
      v9 = sub_2B08680(v7, v8);
      v10 = sub_2B1F810(*(_QWORD *)v6, v9, 0xFFFFFFFF);
      v21 = *(_QWORD *)v6;
      sub_DFB180(*(__int64 **)v6, 1u);
      v11 = sub_DFB120(v21);
      if ( v10 <= v11 )
      {
LABEL_19:
        if ( v11 >> 1 < v10 && v8 )
        {
          _BitScanReverse(&v18, v8);
          v8 = 0x80000000 >> (v18 ^ 0x1F);
        }
        *v19 = v8;
        return v20;
      }
      while ( 1 )
      {
        if ( --v8 )
        {
          _BitScanReverse(&v16, v8);
          v8 = 0x80000000 >> (v16 ^ 0x1F);
        }
        v17 = *(unsigned __int8 *)(v7 + 8);
        if ( (_BYTE)v17 == 17 )
        {
          v12 = v8 * *(_DWORD *)(v7 + 32);
        }
        else
        {
          v12 = v8;
          if ( (unsigned int)(v17 - 17) > 1 )
          {
            v13 = (__int64 *)v7;
            goto LABEL_13;
          }
        }
        v13 = **(__int64 ***)(v7 + 16);
LABEL_13:
        v22 = sub_BCDA70(v13, v12);
        v14 = sub_2B1F810(*(_QWORD *)v6, v22, 0xFFFFFFFF);
        v15 = *(_QWORD *)v6;
        v10 = v14;
        sub_DFB180(*(__int64 **)v6, 1u);
        v11 = sub_DFB120(v15);
        if ( v10 <= v11 )
          goto LABEL_19;
      }
    }
  }
  return v20;
}
