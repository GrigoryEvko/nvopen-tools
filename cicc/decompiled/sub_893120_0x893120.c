// Function: sub_893120
// Address: 0x893120
//
__int64 __fastcall sub_893120(_QWORD **a1, __int64 a2, __int64 a3, _QWORD *a4, int *a5, int a6)
{
  __int64 v8; // rbx
  _QWORD *v9; // r12
  unsigned int v10; // eax
  _QWORD *v11; // r14
  _QWORD *v12; // rdx
  __int64 *v13; // rdi
  int v14; // r14d
  _QWORD *v15; // r12
  __int64 result; // rax
  __int64 *v17; // rdi
  _QWORD *v18; // r15
  _QWORD *v19; // r12
  _QWORD *v20; // r14
  __int64 v21; // rdi
  int v22; // [rsp+4h] [rbp-4Ch]
  _QWORD *v23; // [rsp+8h] [rbp-48h]
  int v24; // [rsp+8h] [rbp-48h]
  _QWORD *v25; // [rsp+10h] [rbp-40h]
  unsigned int v26; // [rsp+18h] [rbp-38h]

  v8 = (__int64)a1;
  v9 = *a1;
  if ( *a1
    && dword_4D047AC
    && (dword_4F04C44 != -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0) )
  {
    v10 = 0;
    if ( unk_4F04C48 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) != 0 && dword_4D047C8 )
    {
      v24 = a6;
      v25 = a4;
      v10 = sub_7D3BE0(qword_4F04C68, a2, dword_4D047C8, a4, a5);
      v9 = *a1;
      a6 = v24;
      a4 = v25;
    }
    v11 = a1;
    v12 = 0;
    while ( 1 )
    {
      if ( v10 && *(_DWORD *)(v11[1] + 44LL) <= v10 )
      {
        if ( v12 )
          *v12 = v9;
        else
          v8 = (__int64)v9;
        v13 = (__int64 *)v11[2];
        if ( v13 )
        {
          v22 = a6;
          v23 = a4;
          v26 = v10;
          sub_725130(v13);
          a6 = v22;
          a4 = v23;
          v10 = v26;
        }
        *v11 = qword_4F601A8;
        qword_4F601A8 = (__int64)v11;
      }
      if ( !v9 )
        break;
      v12 = v11;
      v11 = v9;
      v9 = (_QWORD *)*v9;
    }
  }
  v14 = 0;
  *(_QWORD *)a3 = *(_QWORD *)(v8 + 8);
  *a4 = *(_QWORD *)(v8 + 16);
  v15 = *(_QWORD **)v8;
  if ( *(_QWORD *)v8 )
  {
    if ( *(_BYTE *)(*(_QWORD *)a3 + 80LL) == 19 && !a6 )
    {
      v19 = (_QWORD *)v8;
      v20 = sub_67E020(0x346u, dword_4F07508, a2);
      do
      {
        sub_67E1D0(v20, 839, *(_QWORD *)(*(_QWORD *)(v19[1] + 88LL) + 176LL));
        v19 = (_QWORD *)*v19;
      }
      while ( v19 );
      v21 = (__int64)v20;
      v14 = 1;
      sub_685910(v21, (FILE *)0x347);
      v15 = *(_QWORD **)v8;
    }
    else
    {
      v14 = 1;
    }
  }
  *(_QWORD *)(v8 + 16) = 0;
  result = qword_4F601A8;
  while ( 1 )
  {
    *(_QWORD *)v8 = result;
    qword_4F601A8 = v8;
    if ( !v15 )
      break;
    v17 = (__int64 *)v15[2];
    v18 = (_QWORD *)*v15;
    result = v8;
    if ( v17 )
    {
      sub_725130(v17);
      result = qword_4F601A8;
    }
    v8 = (__int64)v15;
    v15 = v18;
  }
  if ( a5 )
    *a5 = v14;
  return result;
}
