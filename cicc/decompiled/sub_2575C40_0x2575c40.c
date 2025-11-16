// Function: sub_2575C40
// Address: 0x2575c40
//
__int64 __fastcall sub_2575C40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r14
  char v6; // dl
  __int64 v7; // rcx
  __int64 v8; // rax
  unsigned int v10; // esi
  int v11; // eax
  int v12; // eax
  __int64 v13; // [rsp+0h] [rbp-30h] BYREF
  __int64 v14[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( (unsigned __int8)sub_2567ED0(a2, a3, &v13) )
  {
    v5 = v13;
    v6 = 0;
    v7 = *(_QWORD *)a2;
    v8 = *(_QWORD *)(a2 + 8) + 16LL * *(unsigned int *)(a2 + 24);
    goto LABEL_3;
  }
  v10 = *(_DWORD *)(a2 + 24);
  v11 = *(_DWORD *)(a2 + 16);
  v5 = v13;
  ++*(_QWORD *)a2;
  v12 = v11 + 1;
  v14[0] = v5;
  if ( 4 * v12 >= 3 * v10 )
  {
    v10 *= 2;
  }
  else if ( v10 - *(_DWORD *)(a2 + 20) - v12 > v10 >> 3 )
  {
    goto LABEL_6;
  }
  sub_2575960(a2, v10);
  sub_2567ED0(a2, a3, v14);
  v5 = v14[0];
  v12 = *(_DWORD *)(a2 + 16) + 1;
LABEL_6:
  *(_DWORD *)(a2 + 16) = v12;
  if ( (!*(_DWORD *)(v5 + 8) && *(_QWORD *)v5 == -1 || (--*(_DWORD *)(a2 + 20), *(_DWORD *)(v5 + 8) <= 0x40u))
    && *(_DWORD *)(a3 + 8) <= 0x40u )
  {
    *(_QWORD *)v5 = *(_QWORD *)a3;
    *(_DWORD *)(v5 + 8) = *(_DWORD *)(a3 + 8);
  }
  else
  {
    sub_C43990(v5, a3);
  }
  v7 = *(_QWORD *)a2;
  v6 = 1;
  v8 = *(_QWORD *)(a2 + 8) + 16LL * *(unsigned int *)(a2 + 24);
LABEL_3:
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v5;
  *(_QWORD *)(a1 + 24) = v8;
  *(_QWORD *)(a1 + 8) = v7;
  *(_BYTE *)(a1 + 32) = v6;
  return a1;
}
