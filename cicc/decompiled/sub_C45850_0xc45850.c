// Function: sub_C45850
// Address: 0xc45850
//
__int64 __fastcall sub_C45850(__int64 a1, unsigned __int64 **a2, unsigned __int64 a3)
{
  unsigned int v3; // r14d
  unsigned int v4; // eax
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rcx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // [rsp+8h] [rbp-38h] BYREF
  __int64 v11; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-28h]

  v3 = *((_DWORD *)a2 + 2);
  v10 = a3;
  if ( v3 > 0x40 )
  {
    v4 = v3 - sub_C444A0((__int64)a2);
    v5 = ((unsigned __int64)v4 + 63) >> 6;
    if ( !v5 )
      goto LABEL_9;
    v6 = v10;
    if ( v10 == 1 )
    {
      *(_DWORD *)(a1 + 8) = v3;
      sub_C43780(a1, (const void **)a2);
      return a1;
    }
    if ( v4 > 0x40 )
    {
LABEL_7:
      if ( v5 != 1 )
      {
        v12 = v3;
        sub_C43690((__int64)&v11, 0, 0);
        sub_C44DF0((__int64 *)*a2, v5, (__int64 *)&v10, 1u, v11, 0);
        *(_DWORD *)(a1 + 8) = v12;
        *(_QWORD *)a1 = v11;
        return a1;
      }
      v9 = **a2;
      *(_DWORD *)(a1 + 8) = v3;
      sub_C43690(a1, v9 / v6, 0);
      return a1;
    }
    if ( v10 <= **a2 )
    {
      if ( v10 != **a2 )
        goto LABEL_7;
      *(_DWORD *)(a1 + 8) = v3;
      sub_C43690(a1, 1, 0);
    }
    else
    {
LABEL_9:
      *(_DWORD *)(a1 + 8) = v3;
      sub_C43690(a1, 0, 0);
    }
    return a1;
  }
  v8 = (unsigned __int64)*a2;
  *(_DWORD *)(a1 + 8) = v3;
  *(_QWORD *)a1 = v8 / v10;
  return a1;
}
