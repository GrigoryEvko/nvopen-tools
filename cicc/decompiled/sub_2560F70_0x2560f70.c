// Function: sub_2560F70
// Address: 0x2560f70
//
__int64 __fastcall sub_2560F70(__int64 a1, __int64 a2)
{
  __int16 v3; // ax
  __int64 v4; // rdi
  __int64 result; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 i; // r15
  unsigned int v12; // eax
  __int64 v13; // [rsp+8h] [rbp-58h]
  __int64 v14; // [rsp+10h] [rbp-50h] BYREF
  int v15; // [rsp+18h] [rbp-48h]
  __int64 v16; // [rsp+20h] [rbp-40h] BYREF
  int v17; // [rsp+28h] [rbp-38h]

  *(_QWORD *)a1 = &unk_4A170B8;
  v3 = *(_WORD *)(a2 + 16);
  *(_QWORD *)(a1 + 24) = 0;
  *(_WORD *)(a1 + 16) = v3;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 8) = &unk_4A16CD8;
  *(_DWORD *)(a1 + 48) = 0;
  sub_C7D6A0(0, 0, 8);
  v4 = *(unsigned int *)(a2 + 48);
  *(_DWORD *)(a1 + 48) = v4;
  if ( (_DWORD)v4 )
  {
    v6 = sub_C7D670(16 * v4, 8);
    v7 = *(unsigned int *)(a1 + 48);
    v15 = 0;
    *(_QWORD *)(a1 + 32) = v6;
    v8 = v6;
    v9 = *(_QWORD *)(a2 + 40);
    v14 = -1;
    v10 = *(_QWORD *)(a2 + 32);
    *(_QWORD *)(a1 + 40) = v9;
    v17 = 0;
    v16 = -2;
    if ( v7 )
    {
      for ( i = 0; i != v7; ++i )
      {
        v12 = *(_DWORD *)(v10 + 8);
        *(_DWORD *)(v8 + 8) = v12;
        if ( v12 <= 0x40 )
        {
          *(_QWORD *)v8 = *(_QWORD *)v10;
        }
        else
        {
          v13 = v7;
          sub_C43780(v8, (const void **)v10);
          v7 = v13;
        }
        v8 += 16;
        v10 += 16;
      }
    }
    sub_969240(&v16);
    sub_969240(&v14);
  }
  else
  {
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 40) = 0;
  }
  *(_QWORD *)(a1 + 56) = a1 + 72;
  *(_QWORD *)(a1 + 64) = 0x800000000LL;
  if ( *(_DWORD *)(a2 + 64) )
    sub_2560D30((unsigned int *)(a1 + 56), a2 + 56);
  result = *(unsigned __int8 *)(a2 + 200);
  *(_BYTE *)(a1 + 200) = result;
  return result;
}
