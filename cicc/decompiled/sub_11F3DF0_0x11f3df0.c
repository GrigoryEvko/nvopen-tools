// Function: sub_11F3DF0
// Address: 0x11f3df0
//
__int64 *__fastcall sub_11F3DF0(
        __int64 *a1,
        __int64 a2,
        void (__fastcall *a3)(__int64, _QWORD *, __int64),
        __int64 a4,
        void (__fastcall *a5)(__int64, __int64 *, unsigned int *, __int64),
        __int64 a6)
{
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // r11
  __int64 v11; // rcx
  unsigned int v12; // r15d
  unsigned int v13; // ebx
  __int64 v14; // rdx
  unsigned int v15; // eax
  unsigned int v16; // eax
  _BYTE *v17; // rdx
  __int64 v18; // rax
  unsigned int v21; // [rsp+1Ch] [rbp-74h] BYREF
  _QWORD v22[14]; // [rsp+20h] [rbp-70h] BYREF

  v22[5] = 0x100000000LL;
  *a1 = (__int64)(a1 + 2);
  a1[1] = 0;
  *((_BYTE *)a1 + 16) = 0;
  v22[0] = &unk_49DD210;
  v22[6] = a1;
  memset(&v22[1], 0, 32);
  sub_CB5980((__int64)v22, 0, 0, 0);
  a3(a4, v22, a2);
  if ( *(_BYTE *)*a1 == 37 )
    sub_2240CE0(a1, 0, 1);
  v9 = sub_22417D0(a1, 10, 0) + 1;
  if ( v9 > a1[1] )
LABEL_20:
    sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::replace");
  sub_2241130(a1, v9, 0, "\\|", 2);
  v10 = a1[1];
  v21 = 0;
  if ( v10 )
  {
    v11 = *a1;
    v12 = 0;
    v13 = 0;
    v14 = 0;
    v15 = 0;
    do
    {
      v17 = (_BYTE *)(v11 + v14);
      if ( *v17 == 10 )
      {
        *v17 = 92;
        v12 = 0;
        v13 = 0;
        sub_2240FD0(a1, v21 + 1LL, 0, 1, 108);
        v16 = v21;
        v11 = *a1;
        v10 = a1[1];
      }
      else if ( *v17 == 59 )
      {
        v18 = sub_22417D0(a1, 10, v15 + 1);
        a5(a6, a1, &v21, v18);
        v16 = v21;
        v11 = *a1;
        v10 = a1[1];
      }
      else if ( v13 == 80 )
      {
        if ( !v12 )
          v12 = v15;
        if ( v12 > v10 )
          goto LABEL_20;
        sub_2241130(a1, v12, 0, "\\l...", 5);
        v11 = *a1;
        v10 = a1[1];
        v16 = v21 + 3;
        v13 = v21 - v12;
        v21 += 3;
        v12 = 0;
      }
      else
      {
        v16 = v21;
        ++v13;
      }
      v14 = v16 + 1;
      if ( *(_BYTE *)(v11 + v16) == 32 )
        v12 = v16;
      v21 = v16 + 1;
      v15 = v16 + 1;
    }
    while ( v14 != v10 );
  }
  v22[0] = &unk_49DD210;
  sub_CB5840((__int64)v22);
  return a1;
}
