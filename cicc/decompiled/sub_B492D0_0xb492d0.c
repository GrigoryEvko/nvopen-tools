// Function: sub_B492D0
// Address: 0xb492d0
//
__int64 __fastcall sub_B492D0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  unsigned int v5; // eax
  unsigned int v6; // eax
  __int64 v8; // rdi
  unsigned int v9; // eax
  __int64 v10[3]; // [rsp+8h] [rbp-18h] BYREF

  v3 = sub_A747F0((_QWORD *)(a2 + 72), 0, 97);
  if ( v3 )
  {
    v10[0] = v3;
    goto LABEL_3;
  }
  v8 = *(_QWORD *)(a2 - 32);
  if ( v8 )
  {
    if ( !*(_BYTE *)v8 && *(_QWORD *)(v8 + 24) == *(_QWORD *)(a2 + 80) )
    {
      v10[0] = sub_B2D800(v8, 97);
      if ( v10[0] )
      {
LABEL_3:
        v4 = sub_A72AA0(v10);
        v5 = *(_DWORD *)(v4 + 8);
        *(_DWORD *)(a1 + 8) = v5;
        if ( v5 > 0x40 )
        {
          sub_C43780(a1, v4);
          v9 = *(_DWORD *)(v4 + 24);
          *(_DWORD *)(a1 + 24) = v9;
          if ( v9 <= 0x40 )
            goto LABEL_5;
        }
        else
        {
          *(_QWORD *)a1 = *(_QWORD *)v4;
          v6 = *(_DWORD *)(v4 + 24);
          *(_DWORD *)(a1 + 24) = v6;
          if ( v6 <= 0x40 )
          {
LABEL_5:
            *(_QWORD *)(a1 + 16) = *(_QWORD *)(v4 + 16);
LABEL_6:
            *(_BYTE *)(a1 + 32) = 1;
            return a1;
          }
        }
        sub_C43780(a1 + 16, v4 + 16);
        goto LABEL_6;
      }
    }
  }
  *(_BYTE *)(a1 + 32) = 0;
  return a1;
}
