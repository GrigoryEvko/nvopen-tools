// Function: sub_22BE9F0
// Address: 0x22be9f0
//
__int64 __fastcall sub_22BE9F0(int a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rdx
  _QWORD *v8; // rdi
  int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // r15
  __int64 *v12; // rbx
  int v13; // eax
  char v14; // r8
  __int64 result; // rax
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // [rsp+8h] [rbp-68h]
  __int64 v19; // [rsp+8h] [rbp-68h]
  __int64 v20; // [rsp+18h] [rbp-58h]
  unsigned __int64 v21; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v22; // [rsp+28h] [rbp-48h]
  unsigned __int64 v23; // [rsp+30h] [rbp-40h]
  unsigned int v24; // [rsp+38h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 8);
  v8 = *(_QWORD **)v7;
  v9 = *(unsigned __int8 *)(v7 + 8);
  if ( (unsigned int)(v9 - 17) > 1 )
  {
    v11 = sub_BCB2A0(v8);
  }
  else
  {
    BYTE4(v20) = (_BYTE)v9 == 18;
    LODWORD(v20) = *(_DWORD *)(v7 + 32);
    v10 = (__int64 *)sub_BCB2A0(v8);
    v11 = sub_BCE1B0(v10, v20);
  }
  if ( (unsigned __int8)(*(_BYTE *)a3 - 4) > 1u )
  {
    if ( *(_BYTE *)a3 == 3 )
    {
      if ( a1 == 32 )
      {
        v17 = sub_9719A0(0x21u, *(_BYTE **)(a3 + 8), a2, a4, 0, 0);
        if ( v17 && sub_AC30F0(v17) )
          return sub_AD6450(v11);
      }
      else if ( a1 == 33 )
      {
        v16 = sub_9719A0(0x21u, *(_BYTE **)(a3 + 8), a2, a4, 0, 0);
        if ( v16 )
        {
          if ( sub_AC30F0(v16) )
            return sub_AD6400(v11);
        }
      }
    }
    return 0;
  }
  else
  {
    v12 = (__int64 *)(a3 + 8);
    sub_AD8380((__int64)&v21, a2);
    if ( sub_ABB410(v12, a1, (__int64 *)&v21) )
    {
      result = sub_AD6400(v11);
    }
    else
    {
      v13 = sub_B52870(a1);
      v14 = sub_ABB410(v12, v13, (__int64 *)&v21);
      result = 0;
      if ( v14 )
        result = sub_AD6450(v11);
    }
    if ( v24 > 0x40 && v23 )
    {
      v18 = result;
      j_j___libc_free_0_0(v23);
      result = v18;
    }
    if ( v22 > 0x40 )
    {
      if ( v21 )
      {
        v19 = result;
        j_j___libc_free_0_0(v21);
        return v19;
      }
    }
  }
  return result;
}
