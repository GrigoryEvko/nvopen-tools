// Function: sub_1843A50
// Address: 0x1843a50
//
void __fastcall sub_1843A50(_QWORD *a1, __int64 a2)
{
  int v2; // r15d
  int v3; // ebx
  __int64 v4; // rdx
  char v5; // al
  int v6; // r15d
  int v7; // ebx
  __int64 v8; // [rsp+0h] [rbp-40h] BYREF
  int v9; // [rsp+8h] [rbp-38h]
  char v10; // [rsp+Ch] [rbp-34h]

  v8 = a2;
  sub_1843930(a1 + 12, (unsigned __int64 *)&v8);
  if ( (unsigned int)*(_QWORD *)(a2 + 96) )
  {
    v2 = *(_QWORD *)(a2 + 96);
    v3 = 0;
    do
    {
      v9 = v3++;
      v8 = a2;
      v10 = 1;
      sub_1843480(a1, &v8);
    }
    while ( v2 != v3 );
  }
  v4 = **(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL);
  v5 = *(_BYTE *)(v4 + 8);
  if ( v5 )
  {
    if ( v5 == 13 )
    {
      v6 = *(_DWORD *)(v4 + 12);
    }
    else
    {
      if ( v5 != 14 )
      {
        v8 = a2;
        v9 = 0;
        v10 = 0;
        sub_1843480(a1, &v8);
        return;
      }
      v6 = *(_DWORD *)(v4 + 32);
    }
    if ( v6 )
    {
      v7 = 0;
      do
      {
        v9 = v7++;
        v8 = a2;
        v10 = 0;
        sub_1843480(a1, &v8);
      }
      while ( v7 != v6 );
    }
  }
}
