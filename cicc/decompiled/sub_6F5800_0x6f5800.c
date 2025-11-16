// Function: sub_6F5800
// Address: 0x6f5800
//
void __fastcall sub_6F5800(__int64 a1)
{
  _QWORD *v1; // rbx
  __int64 v2; // rax
  __m128i *v3; // r12
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdi
  _QWORD *v14; // rdi
  unsigned int v15; // [rsp+4h] [rbp-CCh] BYREF
  _QWORD *v16; // [rsp+8h] [rbp-C8h] BYREF
  _BYTE v17[18]; // [rsp+10h] [rbp-C0h] BYREF
  char v18; // [rsp+22h] [rbp-AEh]

  v16 = (_QWORD *)a1;
  if ( a1 )
  {
    v1 = (_QWORD *)a1;
    if ( *(_BYTE *)(a1 + 8) != 3 || (sub_72F220(&v16), (v1 = v16) != 0) )
    {
      while ( 1 )
      {
        v2 = v1[6];
        if ( v2 )
        {
          v3 = (__m128i *)(v2 + 8);
          sub_6E1E00(2u, (__int64)v17, 0, 0);
          v18 |= 1u;
          sub_6E18E0((__int64)v3);
          sub_6F5780(v3, (__int64)v17, v4, v5, v6, v7);
          sub_7296C0(&v15);
          v8 = sub_724D50(0);
          sub_695E70((__int64)v3, v8, v9, v10, v11, v12);
          v13 = v15;
          v16[4] = v8;
          sub_729730(v13);
          v14 = (_QWORD *)v16[6];
          sub_6E1940(v14);
          v16[6] = 0;
          sub_6E2B30((__int64)v14, v8);
        }
        else if ( !*((_BYTE *)v1 + 8) )
        {
          v1[4] = sub_8E3240(v1[4], 0);
        }
        v1 = (_QWORD *)*v16;
        v16 = v1;
        if ( !v1 )
          break;
        if ( *((_BYTE *)v1 + 8) == 3 )
        {
          sub_72F220(&v16);
          v1 = v16;
          if ( !v16 )
            break;
        }
      }
    }
  }
}
