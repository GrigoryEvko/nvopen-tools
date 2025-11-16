// Function: sub_CB5B00
// Address: 0xcb5b00
//
void __fastcall sub_CB5B00(int *a1, __int64 a2)
{
  int v2; // edx
  int v3; // eax
  __int64 v4; // rdx
  __int64 v5; // rcx
  _BYTE v6[32]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v7[4]; // [rsp+20h] [rbp-40h] BYREF
  __int16 v8; // [rsp+40h] [rbp-20h]

  *(_QWORD *)a1 = &unk_49DD190;
  if ( a1[12] < 0 )
    goto LABEL_5;
  if ( *((_QWORD *)a1 + 2) != *((_QWORD *)a1 + 4) )
    sub_CB5AE0((__int64 *)a1);
  if ( *((_BYTE *)a1 + 52) && (v3 = sub_C86220(a1[12], a2), v5 = v4, (v2 = v3) != 0) )
  {
    a1[18] = v3;
    *((_QWORD *)a1 + 10) = v5;
  }
  else
  {
LABEL_5:
    v2 = a1[18];
  }
  if ( v2 )
  {
    (*(void (__fastcall **)(_BYTE *))(**((_QWORD **)a1 + 10) + 32LL))(v6);
    v7[0] = "IO failure on output stream: ";
    v7[2] = v6;
    v8 = 1027;
    sub_C64D30((__int64)v7, 0);
  }
  *(_QWORD *)a1 = &unk_49DD388;
  sub_CB5840((__int64)a1);
}
