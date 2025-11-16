// Function: sub_894550
// Address: 0x894550
//
__int64 __fastcall sub_894550(__int64 a1, __int64 a2, _DWORD *a3, _DWORD *a4, _DWORD *a5)
{
  __int64 **v5; // r9
  unsigned int v8; // r14d
  __int64 *v10; // rdi
  int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 **v15; // rcx
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 *v18; // rdi
  unsigned int v19; // r13d
  __int64 v20; // r15
  char v21; // al
  int v22; // [rsp+4h] [rbp-2Ch] BYREF
  __int64 *v23[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = (__int64 **)a2;
  if ( unk_4D049D8 == a1 || unk_4D049D0 == a1 )
  {
    v23[0] = (__int64 *)a2;
    if ( a2 && *(_BYTE *)(a2 + 8) == 3 )
    {
      sub_72F220(v23);
      v5 = (__int64 **)v23[0];
    }
    v10 = *v5;
    v23[0] = v10;
    if ( v10 && *((_BYTE *)v10 + 8) == 3 )
    {
      sub_72F220(v23);
      v10 = v23[0];
    }
    v11 = sub_88D7A0((__int64)v10, a2, (__int64)a3, (__int64)a4, (__int64)a5, (__int64)v5);
    v15 = (__int64 **)v23[0];
    if ( !v11 )
    {
      v16 = v23[0][4];
      v12 = *(unsigned __int8 *)(v16 + 140);
      if ( (_BYTE)v12 == 12 )
      {
        v17 = v23[0][4];
        do
        {
          v17 = *(_QWORD *)(v17 + 160);
          v12 = *(unsigned __int8 *)(v17 + 140);
        }
        while ( (_BYTE)v12 == 12 );
      }
      v8 = 1;
      if ( !(_BYTE)v12 )
        goto LABEL_18;
      v8 = sub_8D2780(v16);
      if ( !v8 )
      {
        if ( a4 )
        {
          a2 = (__int64)a4;
          sub_6851C0(0xAA0u, a4);
        }
        v15 = (__int64 **)v23[0];
        goto LABEL_18;
      }
      v15 = (__int64 **)v23[0];
    }
    v8 = 1;
LABEL_18:
    v18 = *v15;
    v23[0] = v18;
    if ( v18 && *((_BYTE *)v18 + 8) == 3 )
    {
      sub_72F220(v23);
      v18 = v23[0];
    }
    v19 = sub_88D7A0((__int64)v18, a2, v12, (__int64)v15, v13, v14);
    if ( !v19 )
    {
      v20 = v23[0][4];
      v21 = *(_BYTE *)(v20 + 173);
      if ( v21 != 12 )
      {
        if ( v21 )
        {
          if ( (unsigned int)sub_8D2780(*(_QWORD *)(v20 + 128)) )
          {
            if ( sub_620FA0(v20, &v22) < 0 || v22 )
            {
              if ( a5 )
                sub_6851C0(0xA9Fu, a5);
            }
            else
            {
              return v8;
            }
            return v19;
          }
          else
          {
            v8 = 0;
            if ( a5 )
              sub_6851C0(0xAA0u, a5);
          }
        }
      }
    }
    return v8;
  }
  if ( a1 != unk_4D049C8 )
  {
    v8 = 1;
    if ( a1 != unk_4D049C0 )
      return v8;
  }
  return sub_88E560((__int64 *)a2, a3, (__int64)a3, (__int64)a4, (__int64)a5, a2);
}
