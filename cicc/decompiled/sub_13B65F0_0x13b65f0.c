// Function: sub_13B65F0
// Address: 0x13b65f0
//
__int64 *__fastcall sub_13B65F0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdx
  _QWORD *v3; // rax
  unsigned __int64 v4; // r8
  __int64 v5; // rdx
  unsigned int v6; // r13d
  unsigned int v7; // r15d
  __int64 v8; // r12
  unsigned int v9; // ebx
  unsigned int v10; // eax
  char v11; // al
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rax
  _QWORD v16[2]; // [rsp+10h] [rbp-80h] BYREF
  _QWORD v17[2]; // [rsp+20h] [rbp-70h] BYREF
  void *v18; // [rsp+30h] [rbp-60h] BYREF
  _BYTE *v19; // [rsp+38h] [rbp-58h]
  _BYTE *v20; // [rsp+40h] [rbp-50h]
  _BYTE *v21; // [rsp+48h] [rbp-48h]
  int v22; // [rsp+50h] [rbp-40h]
  _QWORD *v23; // [rsp+58h] [rbp-38h]

  v16[0] = v17;
  v16[1] = 0;
  LOBYTE(v17[0]) = 0;
  v22 = 1;
  v21 = 0;
  v20 = 0;
  v19 = 0;
  v18 = &unk_49EFBE0;
  v23 = v16;
  sub_1649960(a2);
  if ( !v2 )
  {
    sub_15537D0(a2, &v18, 0);
    if ( v20 == v21 )
      sub_16E7EE0(&v18, ":", 1);
    else
      *v21++ = 58;
  }
  sub_155C2B0(a2, &v18, 0);
  if ( v21 != v19 )
    sub_16E7BA0(&v18);
  v3 = v23;
  *a1 = (__int64)(a1 + 2);
  sub_13B5790(a1, (_BYTE *)*v3, *v3 + v3[1]);
  if ( *(_BYTE *)*a1 == 10 )
    sub_2240CE0(a1, 0, 1);
  v4 = a1[1];
  if ( v4 )
  {
    v5 = *a1;
    v6 = 0;
    v7 = 0;
    v8 = 0;
    v9 = 0;
    do
    {
      v11 = *(_BYTE *)(v5 + v8);
      if ( v11 == 10 )
      {
        *(_BYTE *)(v5 + v8) = 92;
        v6 = 0;
        v7 = 0;
        sub_2240FD0(a1, v8 + 1, 0, 1, 108);
        v10 = v9;
        v5 = *a1;
        v4 = a1[1];
        ++v9;
      }
      else if ( v11 == 59 )
      {
        v13 = (unsigned int)sub_22417D0(a1, 10, v9 + 1);
        v14 = *a1;
        if ( v13 == a1[1] )
        {
          a1[1] = v8;
          *(_BYTE *)(v14 + v8) = 0;
        }
        else
        {
          sub_2240CE0(a1, v8, v13 - v8);
        }
        v5 = *a1;
        v4 = a1[1];
        v10 = v9 - 1;
      }
      else if ( v7 == 80 )
      {
        if ( !v6 )
          v6 = v9;
        v12 = v6;
        if ( v6 > v4 )
          sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::replace");
        v7 = v9 - v6;
        v6 = 0;
        sub_2241130(a1, v12, 0, "\\l...", 5);
        v10 = v9 + 3;
        v5 = *a1;
        v4 = a1[1];
        v9 += 4;
      }
      else
      {
        v10 = v9;
        ++v7;
        ++v9;
      }
      v8 = v9;
      if ( *(_BYTE *)(v5 + v10) == 32 )
        v6 = v10;
    }
    while ( v9 != v4 );
  }
  sub_16E7BC0(&v18);
  if ( (_QWORD *)v16[0] != v17 )
    j_j___libc_free_0(v16[0], v17[0] + 1LL);
  return a1;
}
