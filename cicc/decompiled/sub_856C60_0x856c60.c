// Function: sub_856C60
// Address: 0x856c60
//
__int64 __fastcall sub_856C60(__int64 a1)
{
  int v1; // ebx
  char v2; // r12
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  char *v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned int v10; // ebx
  __int64 result; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  _QWORD *v16; // rax
  __int64 v17; // rdx
  unsigned int v18; // [rsp+4h] [rbp-1Ch] BYREF
  char *v19; // [rsp+8h] [rbp-18h] BYREF

  v1 = a1;
  a1 = (unsigned int)a1;
  v2 = sub_7AFE70();
  if ( (unsigned int)sub_855EF0((unsigned int)a1, &v18, &v19, v3, v4, v5) )
  {
    if ( v2 )
    {
      if ( v2 == 1 )
      {
        a1 = 2;
        sub_7AFEC0(2);
      }
    }
    else
    {
      a1 = 3;
      sub_7AFEC0(3);
      v16 = qword_4F064B0;
      v17 = qword_4F064B0[12];
      if ( v1 )
        *(_BYTE *)(v17 + 8) |= 4u;
      else
        *(_BYTE *)(v17 + 8) |= 8u;
      v6 = v19;
      *(_QWORD *)(v16[12] + 16LL) = v19;
    }
  }
  v10 = v18;
  result = sub_855540(a1, (__int64)&v18, (__int64)v6, v7, v8, v9);
  if ( !v10 )
    return sub_856950(1u, (__int64)&v18, v12, v13, v14, v15);
  return result;
}
