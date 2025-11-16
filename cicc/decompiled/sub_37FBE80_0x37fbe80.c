// Function: sub_37FBE80
// Address: 0x37fbe80
//
__int64 __fastcall sub_37FBE80(__int16 a1, __int64 a2, unsigned __int16 a3, __int64 a4, __int64 a5, char a6)
{
  int v10; // ebx
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  char v15; // cl
  char v16; // si
  __int64 result; // rax
  __int64 v18; // [rsp+8h] [rbp-68h]
  unsigned __int16 v20; // [rsp+20h] [rbp-50h] BYREF
  __int64 v21; // [rsp+28h] [rbp-48h]
  __int64 v22; // [rsp+30h] [rbp-40h]
  __int64 v23; // [rsp+38h] [rbp-38h]

  v18 = 16LL * (a3 - 1) + 71615648;
  v10 = 2;
  do
  {
    *(_WORD *)a5 = v10;
    *(_QWORD *)(a5 + 8) = 0;
    if ( a3 != (_WORD)v10 )
    {
      v20 = a3;
      v21 = a4;
      if ( a3 )
      {
        if ( a3 == 1 || (unsigned __int16)(a3 - 504) <= 7u )
          BUG();
        v14 = *(_QWORD *)v18;
        v15 = *(_BYTE *)(v18 + 8);
      }
      else
      {
        v11 = sub_3007260((__int64)&v20);
        v13 = v12;
        v22 = v11;
        v14 = v11;
        v23 = v13;
        v15 = v13;
      }
      v16 = byte_444C4A0[16 * v10 - 8];
      if ( !v16 && v15 )
        goto LABEL_8;
      if ( *(_QWORD *)&byte_444C4A0[16 * v10 - 16] < v14 )
      {
        v16 = 0;
LABEL_8:
        result = 729;
LABEL_9:
        if ( (unsigned int)++v10 > 9 )
          return result;
        continue;
      }
    }
    if ( !a6 )
    {
      result = sub_2FE5AE0(a1, a2, *(_DWORD *)a5);
      v16 = (_DWORD)result != 729;
      goto LABEL_9;
    }
    result = sub_2FE5990(a1, a2, *(_DWORD *)a5);
    v16 = (_DWORD)result != 729;
    if ( (unsigned int)++v10 > 9 )
      break;
  }
  while ( !v16 );
  return result;
}
