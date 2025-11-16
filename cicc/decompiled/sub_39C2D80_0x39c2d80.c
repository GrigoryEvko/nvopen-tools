// Function: sub_39C2D80
// Address: 0x39c2d80
//
void __fastcall sub_39C2D80(__int64 a1)
{
  __int64 v2; // r12
  __int64 v3; // rdi
  __int64 v4; // rax
  int v5; // r8d
  int v6; // r9d
  __int64 v7; // rdx
  _QWORD *v8; // rdx
  __int64 v9; // [rsp-20h] [rbp-20h]

  if ( !*(_BYTE *)(a1 + 40) )
  {
    *(_BYTE *)(a1 + 40) = 1;
    if ( (unsigned __int8)sub_39C2C50(*(_QWORD *)a1, *(_QWORD *)(a1 + 8)) )
    {
      v2 = *(_QWORD *)(a1 + 16);
      v3 = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(v2 + 32) = v3;
      v4 = sub_1E16510(v3);
      if ( v4 && (unsigned int)((__int64)(*(_QWORD *)(v4 + 32) - *(_QWORD *)(v4 + 24)) >> 3) )
      {
        v7 = *(unsigned int *)(v2 + 48);
        if ( (unsigned int)v7 >= *(_DWORD *)(v2 + 52) )
        {
          v9 = v4;
          sub_16CD150(v2 + 40, (const void *)(v2 + 56), 0, 16, v5, v6);
          v7 = *(unsigned int *)(v2 + 48);
          v4 = v9;
        }
        v8 = (_QWORD *)(*(_QWORD *)(v2 + 40) + 16 * v7);
        *v8 = 0;
        v8[1] = v4;
        ++*(_DWORD *)(v2 + 48);
      }
      *(_DWORD *)(*(_QWORD *)(a1 + 16) + 24LL) = *(_QWORD *)(a1 + 32);
    }
  }
}
