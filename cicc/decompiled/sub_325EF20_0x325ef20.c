// Function: sub_325EF20
// Address: 0x325ef20
//
__int64 __fastcall sub_325EF20(__int64 a1, unsigned int *a2, unsigned int a3, __int64 a4, __int64 a5)
{
  unsigned int v6; // esi
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 result; // rax
  _QWORD *v12; // r11
  unsigned int v13; // ecx
  _QWORD *v14; // rdx
  _QWORD *v15; // rax
  _QWORD *v16; // rax
  _QWORD *v17; // rdi
  unsigned int v18; // esi
  __int64 v19; // rdx
  __int64 v20; // rax
  _QWORD *v21; // rdx
  _QWORD *v22; // rax

  v6 = *a2;
  if ( !v6 )
  {
LABEL_4:
    if ( v6 == a3 )
    {
      v17 = (_QWORD *)(16LL * v6 + a1);
      *v17 = a4;
      v17[1] = a5;
      return v6 + 1;
    }
    else
    {
      v12 = (_QWORD *)(a1 + 16LL * v6);
      if ( a5 == *v12 )
      {
        *v12 = a4;
        return a3;
      }
      else
      {
        result = 9;
        if ( a3 != 8 )
        {
          v13 = a3 - 1;
          do
          {
            v14 = (_QWORD *)(a1 + 16LL * v13);
            v15 = (_QWORD *)(a1 + 16LL * (v13 + 1));
            *v15 = *v14;
            v15[1] = v14[1];
            LODWORD(v15) = v13--;
          }
          while ( v6 != (_DWORD)v15 );
          *v12 = a4;
          v12[1] = a5;
          return a3 + 1;
        }
      }
    }
    return result;
  }
  v9 = v6 - 1;
  v10 = a1 + 16 * v9;
  if ( *(_QWORD *)(v10 + 8) != a4 )
  {
    result = 9;
    if ( v6 == 8 )
      return result;
    goto LABEL_4;
  }
  *a2 = v9;
  if ( v6 != a3 && (v16 = (_QWORD *)(a1 + 16LL * v6), *v16 == a5) )
  {
    v18 = v6 + 1;
    for ( *(_QWORD *)(v10 + 8) = v16[1]; a3 != v18; v22[1] = v21[1] )
    {
      v19 = v18;
      v20 = v18++ - 1;
      v21 = (_QWORD *)(a1 + 16 * v19);
      v22 = (_QWORD *)(a1 + 16 * v20);
      *v22 = *v21;
    }
    return a3 - 1;
  }
  else
  {
    *(_QWORD *)(v10 + 8) = a5;
    return a3;
  }
}
