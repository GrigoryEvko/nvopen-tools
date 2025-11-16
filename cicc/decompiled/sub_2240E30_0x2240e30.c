// Function: sub_2240E30
// Address: 0x2240e30
//
void __fastcall sub_2240E30(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v4; // rcx
  _BYTE *v5; // r12
  _BYTE *v6; // rbp
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rsi
  __int64 v9; // rax
  _BYTE *v10; // rdi
  _BYTE *v11; // r12
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14[4]; // [rsp+8h] [rbp-20h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 8);
  v14[0] = a2;
  if ( a2 < v4 )
  {
    v14[0] = v4;
    v2 = v4;
  }
  v5 = *(_BYTE **)a1;
  v6 = (_BYTE *)(a1 + 16);
  if ( a1 + 16 == *(_QWORD *)a1 )
    v7 = 15;
  else
    v7 = *(_QWORD *)(a1 + 16);
  if ( v7 != v2 )
  {
    v8 = 15;
    if ( v7 <= 0xF )
      v8 = v7;
    if ( v8 >= v2 )
    {
      if ( v6 != v5 )
      {
        if ( v4 )
        {
          if ( v4 != -1 )
            memcpy((void *)(a1 + 16), *(const void **)a1, v4 + 1);
        }
        else
        {
          *(_BYTE *)(a1 + 16) = *v5;
        }
        j___libc_free_0((unsigned __int64)v5);
        *(_QWORD *)a1 = v6;
      }
    }
    else
    {
      v9 = sub_22409D0(a1, v14, v7);
      v10 = *(_BYTE **)a1;
      v11 = (_BYTE *)v9;
      v12 = *(_QWORD *)(a1 + 8);
      if ( v12 )
      {
        if ( v12 != -1 )
        {
          memcpy(v11, *(const void **)a1, v12 + 1);
          v10 = *(_BYTE **)a1;
        }
      }
      else
      {
        *v11 = *v10;
        v10 = *(_BYTE **)a1;
      }
      if ( v6 != v10 )
        j___libc_free_0((unsigned __int64)v10);
      v13 = v14[0];
      *(_QWORD *)a1 = v11;
      *(_QWORD *)(a1 + 16) = v13;
    }
  }
}
