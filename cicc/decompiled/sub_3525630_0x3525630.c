// Function: sub_3525630
// Address: 0x3525630
//
__int64 __fastcall sub_3525630(
        __int64 a1,
        __int64 a2,
        void (__fastcall *a3)(__int64, _QWORD *, __int64),
        __int64 a4,
        void (__fastcall *a5)(__int64, __int64, unsigned int *, char *),
        __int64 a6)
{
  char *v9; // rax
  unsigned __int64 v10; // rcx
  size_t v11; // rsi
  unsigned __int64 v12; // r11
  __int64 v13; // rcx
  unsigned int v14; // r15d
  unsigned int v15; // ebx
  __int64 v16; // rdx
  unsigned int v17; // eax
  unsigned int v18; // eax
  _BYTE *v19; // rdx
  char *v20; // rax
  unsigned int v23; // [rsp+1Ch] [rbp-74h] BYREF
  _QWORD v24[14]; // [rsp+20h] [rbp-70h] BYREF

  v24[5] = 0x100000000LL;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v24[0] = &unk_49DD210;
  v24[6] = a1;
  memset(&v24[1], 0, 32);
  sub_CB5980((__int64)v24, 0, 0, 0);
  a3(a4, v24, a2);
  if ( **(_BYTE **)a1 == 37 )
    sub_2240CE0((__int64 *)a1, 0, 1);
  v9 = sub_22417D0((__int64 *)a1, 10, 0);
  v10 = *(_QWORD *)(a1 + 8);
  v11 = (size_t)(v9 + 1);
  if ( (unsigned __int64)(v9 + 1) > v10 )
LABEL_21:
    sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", "basic_string::replace", v11, v10);
  sub_2241130((unsigned __int64 *)a1, v11, 0, "\\|", 2u);
  v12 = *(_QWORD *)(a1 + 8);
  v23 = 0;
  if ( v12 )
  {
    v13 = *(_QWORD *)a1;
    v14 = 0;
    v15 = 0;
    v16 = 0;
    v17 = 0;
    do
    {
      v19 = (_BYTE *)(v13 + v16);
      if ( *v19 == 10 )
      {
        *v19 = 92;
        v14 = 0;
        v15 = 0;
        sub_2240FD0((unsigned __int64 *)a1, v23 + 1LL, 0, 1u, 108);
        v18 = v23;
        v13 = *(_QWORD *)a1;
        v12 = *(_QWORD *)(a1 + 8);
      }
      else if ( *v19 == 59 )
      {
        v20 = sub_22417D0((__int64 *)a1, 10, v17 + 1);
        a5(a6, a1, &v23, v20);
        v18 = v23;
        v13 = *(_QWORD *)a1;
        v12 = *(_QWORD *)(a1 + 8);
      }
      else if ( v15 == 80 )
      {
        if ( !v14 )
          v14 = v17;
        v11 = v14;
        if ( v14 > v12 )
        {
          v10 = v12;
          goto LABEL_21;
        }
        sub_2241130((unsigned __int64 *)a1, v14, 0, "\\l...", 5u);
        v13 = *(_QWORD *)a1;
        v12 = *(_QWORD *)(a1 + 8);
        v18 = v23 + 3;
        v15 = v23 - v14;
        v23 += 3;
        v14 = 0;
      }
      else
      {
        v18 = v23;
        ++v15;
      }
      v16 = v18 + 1;
      if ( *(_BYTE *)(v13 + v18) == 32 )
        v14 = v18;
      v23 = v18 + 1;
      v17 = v18 + 1;
    }
    while ( v16 != v12 );
  }
  v24[0] = &unk_49DD210;
  sub_CB5840((__int64)v24);
  return a1;
}
