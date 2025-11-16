// Function: sub_39A6830
// Address: 0x39a6830
//
void __fastcall sub_39A6830(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v5; // rbx
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 v8; // [rsp+0h] [rbp-40h]

  if ( a3 )
  {
    v3 = *(unsigned int *)(a3 + 8);
    if ( (unsigned int)v3 > 1 )
    {
      v8 = *(unsigned int *)(a3 + 8);
      v5 = 1;
      while ( 1 )
      {
        v6 = *(_QWORD *)(a3 + 8 * (v5 - v3));
        if ( !v6 )
          break;
        v7 = sub_39A5A90((__int64)a1, 5, a2, 0);
        sub_39A6760(a1, v7, v6, 73);
        if ( (*(_BYTE *)(v6 + 28) & 0x40) == 0 )
          goto LABEL_4;
        ++v5;
        sub_39A34D0((__int64)a1, v7, 52);
        if ( v8 == v5 )
          return;
LABEL_5:
        v3 = *(unsigned int *)(a3 + 8);
      }
      sub_39A5A90((__int64)a1, 24, a2, 0);
LABEL_4:
      if ( v8 == ++v5 )
        return;
      goto LABEL_5;
    }
  }
}
