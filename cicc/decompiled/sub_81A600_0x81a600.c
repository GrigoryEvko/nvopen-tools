// Function: sub_81A600
// Address: 0x81a600
//
__int64 __fastcall sub_81A600(unsigned __int64 a1, __int64 a2, __int64 a3, int a4)
{
  unsigned int v4; // r14d
  unsigned __int64 v6; // r12
  _QWORD *v7; // rdx
  unsigned __int64 v8; // rax
  _QWORD *v9; // r15
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rax
  _QWORD *i; // rdx
  unsigned __int64 v14; // rax
  _QWORD *v15; // rdx
  unsigned __int64 v16; // rax
  _QWORD *v17; // rdx
  unsigned __int64 *v18; // rcx
  unsigned __int64 v19; // rax
  __int64 v21; // rdx
  unsigned __int64 *v22; // rcx
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rax

  v4 = 0;
  if ( a1 != a3 )
  {
    v6 = a2 + 1;
    v7 = (_QWORD *)unk_4F06458;
    if ( unk_4F06458 )
    {
      do
      {
        while ( 1 )
        {
          v8 = v7[1];
          if ( v8 != 0 && v8 >= a1 && v8 < v6 )
            break;
          v7 = (_QWORD *)*v7;
          if ( !v7 )
            goto LABEL_8;
        }
        v4 = 1;
        v7[1] = a3 + v8 - a1;
        v7 = (_QWORD *)*v7;
      }
      while ( v7 );
    }
LABEL_8:
    if ( a4 )
    {
      v9 = (_QWORD *)qword_4F06440;
      if ( qword_4F06440 )
      {
        do
        {
          v10 = v9[2];
          if ( v10 >= a1 && v10 != 0 && v10 < v6 )
          {
            sub_7AED90((__int64)v9);
            v24 = v9[2];
            if ( v24 >= a1 && v24 != 0 && v24 < v6 )
            {
              v4 = 1;
              v9[2] = a3 + v24 - a1;
            }
            sub_7AED40((__int64)v9);
          }
          v11 = v9[7];
          if ( v11 >= a1 && v11 != 0 && v11 < v6 )
          {
            v4 = 1;
            v9[7] = a3 + v11 - a1;
          }
          v12 = v9[8];
          if ( v12 >= a1 && v12 != 0 && v12 < v6 )
          {
            v4 = 1;
            v9[8] = a3 + v12 - a1;
          }
          for ( i = (_QWORD *)v9[13]; i; i = (_QWORD *)*i )
          {
            while ( 1 )
            {
              v14 = i[1];
              if ( v14 != 0 && v14 >= a1 && v14 < v6 )
                break;
              i = (_QWORD *)*i;
              if ( !i )
                goto LABEL_24;
            }
            v4 = 1;
            i[1] = a3 + v14 - a1;
          }
LABEL_24:
          v9 = (_QWORD *)*v9;
        }
        while ( v9 );
      }
    }
    v15 = (_QWORD *)qword_4F194E0;
    if ( qword_4F194E0 )
    {
      do
      {
        while ( 1 )
        {
          v16 = v15[4];
          if ( v16 != 0 && v16 >= a1 && v16 < v6 )
            break;
          v15 = (_QWORD *)*v15;
          if ( !v15 )
            goto LABEL_31;
        }
        v4 = 1;
        v15[4] = a3 + v16 - a1;
        v15 = (_QWORD *)*v15;
      }
      while ( v15 );
    }
LABEL_31:
    if ( (unsigned __int64)qword_4F06460 < v6 && qword_4F06460 != 0 && (unsigned __int64)qword_4F06460 >= a1 )
    {
      v4 = 1;
      qword_4F06460 = &qword_4F06460[a3 - a1];
    }
    if ( (unsigned __int64)qword_4F06420 < v6 && qword_4F06420 != 0 && (unsigned __int64)qword_4F06420 >= a1 )
    {
      v4 = 1;
      qword_4F06420 = &qword_4F06420[a3 - a1];
    }
    if ( (unsigned __int64)qword_4F06410 < v6 && qword_4F06410 != 0 && (unsigned __int64)qword_4F06410 >= a1 )
    {
      v4 = 1;
      qword_4F06410 = &qword_4F06410[a3 - a1];
    }
    if ( qword_4F06408 < v6 && qword_4F06408 != 0 && qword_4F06408 >= a1 )
    {
      v4 = 1;
      qword_4F06408 = a3 + qword_4F06408 - a1;
    }
    if ( qword_4F194D0 < v6 && qword_4F194D0 != 0 && qword_4F194D0 >= a1 )
    {
      v4 = 1;
      qword_4F194D0 = a3 + qword_4F194D0 - a1;
    }
    v17 = (_QWORD *)unk_4D03BE0;
    if ( unk_4D03BE0 )
    {
      do
      {
        while ( 1 )
        {
          v18 = (unsigned __int64 *)v17[1];
          v19 = *v18;
          if ( *v18 != 0 && *v18 >= a1 && v19 < v6 )
            break;
          v17 = (_QWORD *)*v17;
          if ( !v17 )
            goto LABEL_52;
        }
        v4 = 1;
        *v18 = a3 + v19 - a1;
        v17 = (_QWORD *)*v17;
      }
      while ( v17 );
    }
LABEL_52:
    if ( qword_4F06498 == a1 && *(int *)&word_4F06480 > 0 )
    {
      v21 = 0;
      do
      {
        v22 = &qword_4F06488[v21];
        v23 = *v22;
        if ( *v22 >= a1 && *v22 != 0 && v23 < v6 )
        {
          v4 = 1;
          *v22 = a3 + v23 - a1;
        }
        ++v21;
      }
      while ( *(int *)&word_4F06480 > (int)v21 );
    }
  }
  return v4;
}
